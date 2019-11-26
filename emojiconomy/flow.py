from networkx.algorithms.flow import maximum_flow
from networkx.algorithms.dag import topological_sort

import math
import itertools

def amount_consumed( g ):
    sinks = [ n for n,t in g.nodes.data( 'tag' ) if t == 'sink']
    consumed = {}
    for i,j,e in g.in_edges( sinks, data=True ):
        # FIXME: use tag
        consumed[ g.nodes[i]['text'] ] = g.edges[(i,j)]['flow'] 
    return consumed
    
              
def linear_metric( consumables ):
    return sum( consumables.values() )

# Scaling by 0.004 means we're at 0.96 utility at 1000 units
def sigmoid_1000( x ):
    return 2.0 * math.exp( 0.004 * x ) / ( math.exp( 0.004 * x ) + 1 ) - 1.0

def sigmoid_metric( consumables ):
    return sum( sigmoid_1000( x ) for x in consumables.values() )


def project_onto_standard_simplex( y ):
    """Find the nearest neighbor of y on the |y|-element standard simplex
    x_1 + ... + x_n = 1
  
    See Yunmei Chen and Xiaojing Ye, "Projection Onto a Simplex", 
    https://arxiv.org/abs/1101.6081
    """
    n = len( y )
    y_s = sorted( y, reverse=True )

    # Sum the i largest y's.
    # Compute t_i = (sum) / (i)
    # if t_i >= next-smallest y, break
    #
    # In the paper the order is reversed, so we get 
    # t_i = ( -1 + sum_{j=i+1}^{n} y_j  ) / (n-i)
    #
    # The last check of the loop doesn't matter, so we can fill in any
    # arbitrary value for y_next.
    sum_y = 0
    for i, y_i, y_next in zip( range( 1, n+1 ), y_s, y_s[1:] + [0.0] ):
        sum_y += y_i
        t = (sum_y - 1) / i
        if t >= y_next:
            break

    return [ max( 0, y_i - t ) for y_i in y ]
                

class FlowModel(object):
    def __init__( self, g, metric = None):
        self.graph = g
        if metric is None:
            self.metric = sigmoid_metric
        else:
            self.metric = metric
        self.initial_goods = []
        self.simplices = []
        self.fixed_flows = {}
        
    def set_fixed_flow( self, src, dst, flow ):
        # Remove the given edge as a parameter.
        for i,s in enumerate( self.simplices ):
            if (src,dst) in s:
                s.remove( (src,dst) )
                if len( s ) <= 1:
                    del self.simplices[i]
                self.num_parameters = sum( len( s ) for s in self.simplices )
                break
            
        self.fixed_flows[ (src,dst) ] = flow
        
    def set_initial_goods( self, amount = 1000 ):
        sources = [ n for n,t in self.graph.nodes.data( 'tag' )
                    if t == 'source']
        for n in self.initial_goods:
            for s in sources:
                self.set_fixed_flow( s, n, amount )
        
    def find_parameters( self ):
        # At choice points we must assign inputs to outputs.
        # input >= out_1 + out_2 + ... + out_k
        #
        # In a relaxed model, each can be viewed as a simplex that sums to 1.
        #
        for n in self.graph:
            if 'tag' not in self.graph.nodes[n]:
                continue
            
            if self.graph.nodes[n]['tag'] == 'source':
                for i,j in self.graph.out_edges( n ):
                    self.initial_goods.append( j )
            elif self.graph.nodes[n]['category'] == 'factory':
                pass
            elif self.graph.nodes[n]['category'] == 'process':
                pass
            elif self.graph.out_degree(n) > 1:
                simplex = list( self.graph.out_edges(n) )
                self.simplices.append( simplex )

        self.num_parameters = sum( len( s ) for s in self.simplices )

    def simplex_iter( self, x ):
        pos = 0
        for simplex in self.simplices:    
            p_size = len( simplex )
            p = x[pos:pos+p_size]
            pos += p_size
            yield ( simplex, p )
        
    def relaxed_start( self ):
        p = []
        for s in self.simplices:
            x = 1.0 / len( s )
            p.extend( [x] * ( len( s ) ) )
        return p

    def project_onto_simplices( self, x_0 ):
        """Project a relaxed solution onto its nearest L2-norm neighbor
        that obeys the simplex constraints.
        """
        z = []
        for simplex, x in self.simplex_iter( x_0 ):
            z.extend( project_onto_standard_simplex( x ) )
            
        return z

    def handle_branch_node( self,
                            g,
                            n,
                            relaxed ):
        g.nodes[n]['quantity'] += \
            sum( f for i, j, f in g.in_edges(n,data='flow') )
        available = g.nodes[n]['quantity']

        min_flow = g.out_edges(n,data="min_flow")
        min_flow_required = sum( f for i,j,f in min_flow )
        # Might go negative.
        available -= min_flow_required
        outputs = g.out_edges(n,data=True)
        if relaxed:                    
            # Interpret as 0--1 simplex
            for (i,j,attr) in outputs:
                w = attr['weight']
                if attr['min_flow'] == 0:
                    g.edges[(i,j)]['flow'] = available * w
        elif len( outputs ) == 1:
            for (i,j,attr) in outputs:
                if attr['min_flow'] == 0:
                    if available < 0:
                        g.graph['negative_flows'] = True
                    g.edges[(i,j)]['flow'] = available
                elif j == 'EXP':
                    # Don't export more than is available.
                    # FIXME: come up with a less hackish way to do this.
                    if available < 0:
                        print( "WARNING: export took us negative." )
                        g.graph['negative_flows'] = True
            g.nodes[n]['slack'] = 0
        else:      
            # Interpret as item counts, if feasible
            # min_flow already removed from available
            committed = sum( attr['weight'] for i,j,attr in outputs )
            total = committed - min_flow_required
            # print( "available", available, "total output", total )
            if total <= available:
                for (i,j,attr) in outputs:
                    if attr['min_flow'] == 0:
                        g.edges[(i,j)]['flow'] = attr['weight']
                        # print( (i,j), "flow", attr['weight'] )
                g.nodes[n]['slack'] = available - total
            else:
                # Infeasible, scale down
                if total != 0:
                    for (i,j,attr) in outputs:
                        if attr['min_flow'] == 0:
                            rounded_flow = available * attr['weight'] // total
                            available -= rounded_flow
                            g.edges[(i,j)]['flow'] = rounded_flow
                            if rounded_flow < 0:
                                g.graph['negative_flows'] = True

                            # print( (i,j), "rounded flow", rounded_flow )
                g.nodes[n]['slack'] = available
        
    def compute_flow( self,
                      parameters,
                      relaxed = False ):
        g = self.graph.copy()
        
        # Does graph contain any negative flows?
        g.graph['negative_flows'] = False

        for n in g.nodes:
            g.nodes[n]['quantity'] = 0
            g.nodes[n]['slack'] = 0

        for i,j in g.edges:
            g[i][j]['flow'] = 0
            g[i][j]['min_flow'] = 0
            if relaxed:
                g[i][j]['weight'] = 1.0
            else:
                # Infinity!
                g[i][j]['weight'] = 1000000
            
        pos = 0
        for simplex, x in self.simplex_iter( parameters ):
            for e,weight in zip( simplex, x ):
                g.edges[e]['weight'] = weight

        for edge,amount in self.fixed_flows.items():
            g.edges[edge]['flow'] = amount
            g.edges[edge]['min_flow'] = amount
            g.edges[edge]['weight'] = amount
                
        for n in topological_sort( g ):
            if self.graph.nodes[n]['tag'] == 'source':
                # Show flows as if they came from source.
                continue
            # If node is a multi-output processing plant, its outgoing
            # edges are tagged '=' and the input should be divided equally.
            elif self.graph.nodes[n]['category'] == 'process':
                g.nodes[n]['quantity'] += \
                    sum( f for i,j,f in g.in_edges(n,data='flow') )
                outputs = g.out_edges( n )
                num_outputs = len( outputs )
                if relaxed:
                    units = g.nodes[n]['quantity'] / num_outputs
                else:
                    # Integer distribution
                    units = g.nodes[n]['quantity'] // num_outputs
                    
                for o in outputs:
                    g.edges[o]['flow'] = units
                    
            # If node is a factory, its outgoing edge is tagged 'min'
            # and the output is 2*max of inputs.
            elif self.graph.nodes[n]['category'] == 'factory':
                min_input = min( f for i,j,f
                                 in g.in_edges(n,data='flow') )
                produced = min_input * g.in_degree( n )
                g.nodes[n]['quantity'] += produced
                outputs = g.out_edges( n )
                assert len( outputs ) <= 1
                for o in outputs:
                    g.edges[o]['flow'] = g.nodes[n]['quantity']
                    
            # Otherwise, we have a choice and apply the weights from the
            # parameters, or no choice and send the entire output.
            else:
                self.handle_branch_node( g, n, relaxed )
                            
        c = amount_consumed( g )
        return g, c, self.metric( c )

    def gradient( self, p, m = None ):
        if m is None:
            _, _, m = self.compute_flow( p, relaxed=True )

        grad = [ 0.0 for _ in range( len( p ) )  ]
        delta = 0.001  # FIXME
        for i in range( len( p ) ):
            p_prime = list( p )
            p_prime[i] += delta
            _, _, m_prime = self.compute_flow( p_prime, relaxed=True )
            grad[i] = ( m_prime - m ) / delta
        return grad
    
    def unrelaxed_weights( self, x ):
        flows, _, m = self.compute_flow( x,
                                         relaxed=True )
        x_int = []
        for simplex, _ in self.simplex_iter( x ):
            x_int.extend( int( flows.edges[e]['flow'] ) for e in simplex )
        return x_int
        

def round( x ):
    return int( x * 100 + 0.5 ) / 100.0

def smallest_step( x, g ):
    return abs( 0.01 / g )
    
def largest_step( x, g ):
    if g > 0:
        # x + step * g <= 1
        # step <= (1-x)/g
        return (1 - x) / g
    else:
        # 0 <= x + step * g
        # -x <= step * g
        # step <= -x / g
        return -x / g

def flatten( list_of_lists ):
    return [item for sublist in list_of_lists for item in sublist]

def pairwise_sum( x, delta ):
    return [ x_i + delta_i for x_i, delta_i in zip( x, delta ) ]

def step_indicator( x, y ):
    ret = ""
    for x_i, y_i in zip( x, y ):
        if x_i - y_i > 0:
            ret += "+"
        elif x_i == y_i:
            ret += "0"
        else:
            ret += "-"
    return ret

def in_range( x ):
    return min( 0.0 <= x_i and x_i <= 1 for x_i in x )

def all_zeros( x ):
    return not max( x_i > 0.0 for x_i in x )

def nonnegative( x ):
    return min( x_i >= 0 for x_i in x )

def simplex_increase( simplex_size ):
    for i in range( simplex_size ):
        delta = [0] * simplex_size
        delta[i] = 1
        yield delta
        
def simplex_stable( simplex_size ):
    yield [0] * simplex_size
    for i, j in itertools.combinations( range( simplex_size ), r = 2 ):
        delta = [0] * simplex_size
        delta[i] = 1
        delta[j] = -1
        yield delta
            
def step_solver( f, x ):
    flow, consumption, y = f.compute_flow( x )
    print( "Start:", y, x )

    visited = set()
    visited.add( tuple(x) )
    
    while True:
        # If there is slack in the simplex, only look at increases
        # otherwise, only permit balanced operations.
        simplexSteps = []
        
        for simplex, outputs in f.simplex_iter( x ):
            sourceNode = simplex[0][0]
            available = flow.nodes[sourceNode]['quantity']
            if sum( outputs ) < available:
                simplexSteps.append( list(
                    simplex_increase( len( simplex ) ) ) )
            else:
                simplexSteps.append( list(
                    simplex_stable( len( simplex ) ) ) )
                                     
        allSteps = itertools.product( *simplexSteps )
        next_xs = [ pairwise_sum( x, flatten(d) ) for d in allSteps ]
        
        neighborhood = [ (x_p, f.compute_flow( x_p )) for x_p in next_xs
                         if not tuple(x_p) in visited ]

        if debug_step:
            print( len( neighborhood ), "neighbors" )
        visited.update( tuple(x_p) for x_p, _ in neighborhood )

        if len( neighborhood ) == 0:
            break

        x_p, (f_p, c_p, y_p) = max( neighborhood, key = lambda t : t[1][2] )
        if debug_step:
            print( "Best: ", y_p, x_p ) 
        if y_p <= y:
            if False:
                for x_p, (f_p, c_p, y_p) in neighborhood:
                    print( y_p, x_p, step_indicator( x_p, x ) )
            break
        else:
            flow = f_p
            consumption = c_p
            y = y_p
            x = x_p
            
    return flow, consumption, y, x       

def augmenting_paths( flow, good, amount, memo ):
    key = ( good, amount )
    if key in memo:
        return memo[key]
    
    # We always have two options:
    # Pay for the change locally by taking away from a different consumer
    # Or, propogate the change upward.
    # FIXME: or a combination of the two?
    
    # If enough to satisfy already, don't need to propogate changes
    amount -= flow.nodes[good]['slack']
    if amount <= 0:
        return [[]]

    choices = [[]]

    if flow.nodes[good]['category'] == 'factory':
        # Augment all inputs, by 1/2 the amount
        inputs = []
        for i,j,e in flow.in_edges( [good], data=True ):
            needed = (amount+1) // 2
            inputs.append( [a + [(i,j,needed)]
                            for a in augmenting_paths( flow, i, needed, memo ) ] )
        assert len( inputs ) == 2
        # Combination of all ways to increase input A, with all
        # ways to simultanously increase input B.
        for a in inputs[0]:
            for b in inputs[1]:
                choices.append( a + b )
    elif flow.nodes[good]['category'] == 'process':
        # Augment input by 2x the amount
        inputs = []
        for i,j,e in flow.in_edges( [good], data=True ):
            needed = amount * 2
            inputs.append( [ a + [(i,j,needed)]
                             for a in augmenting_paths( flow, i, needed, memo ) ] )
        assert len( inputs ) == 1
        choices.extend( inputs[0] )
    elif flow.nodes[good]['category'] == 'source':
        pass
    else:
        # Augment any input by 1x
        for i,j,e in flow.in_edges( [good], data=True ):
            for input_augmented in augmenting_paths( flow, i, amount, memo ):
                choices.append( input_augmented + [(i,j,amount)] )

    # print( "Augment", good, "by", amount, "choices", choices )

    memo[key] = choices
    return choices

def possible_decreases( simplex, needed ):
    if len( simplex ) == 0:
        if needed == 0:
            yield []
        return
        
    if simplex[0] > 0:
        for d in possible_decreases( simplex[1:], needed ):
            yield [0] + d
    else:
        # We can increase at this position
        for i in range( needed + 1 ):
            for d in possible_decreases( simplex[1:], needed - i ):
                yield [-i] + d            
               
def augmenting_path_solver( f , x, verbose = False ):
    flow, consumption, y = f.compute_flow( x )
    if verbose:
        print( "Start:", y, x )

    while True:
        # Attempt to increase consumption of a good by 1.
        # FIXME: use 2?
        paths = []
        sinks = [ n for n,t in flow.nodes.data( 'tag' ) if t == 'sink']
        memo = {}
        for i,j,e in flow.in_edges( sinks, data=True ):
            for p in augmenting_paths( flow, i, 1, memo ):
                p.append( (i,j,1) )
                paths.append( p )

        # Not every augmenting path leads to a different set of parameters,
        # deduplicate them in this dictionary (which stores one of the paths).
        neighbors = {}
        
        for p in paths:
            # there's more than one way to do it!  Keep a list, which
            # we'll build up in parallel, simplex by simplex
            x_for_path = [[]]
            
            # print( "Path", p )
            for simplex, x_s in f.simplex_iter( x ):
                source = simplex[0][0]
                
                # desired increases within this simplex
                increases = [
                    sum( a for (i,j,a) in p if (i,j) == e )
                    for e in simplex
                ]
                decrease_needed = \
                    sum( increases ) - flow.nodes[source]['slack']
                #print( "Simplex", simplex, "Desired increase: ", increases,
                #       "Decrease needed:", decrease_needed )
                if decrease_needed <= 0:
                    x_for_path = [
                        prev_x + list( z + inc
                                       for z, inc in zip( x_s, increases ) )
                        for prev_x in x_for_path
                    ]
                else:
                    # Extend every partial x with new coefficients
                    # that have the desired increases
                    # and every possible set of decreases
                    x_for_path = [
                        prev_x + list( z + inc + dec
                                       for z, inc, dec in zip( x_s, increases, decreases ) )
                        for decreases in possible_decreases( increases, decrease_needed )
                        for prev_x in x_for_path
                    ]

            for x_p in x_for_path:
                if x_p != x and nonnegative( x_p ):
                    neighbors[tuple(x_p)] = p
        
        neighborhood = [ (x_p, f.compute_flow( x_p ),p)
                         for (x_p,p) in neighbors.items() ]

        if verbose:
            print( "num neighbors:", len( neighborhood ) )

        if len( neighborhood ) == 0:
            break

        x_p, (f_p, c_p, y_p), best_path = \
            max( neighborhood, key = lambda t : t[1][2] )
        if verbose:
            print( "Best:", y_p, x_p ) 
            print( "Best path: ", best_path ) 

        if y_p <= y:
            if verbose:
                for n in neighborhood:
                    print( "X=",n[0], "Y=", n[1][2], "P=", n[2] )
                    # print( n[1][1] )
            break
        else:
            flow = f_p
            consumption = c_p
            y = y_p
            x = x_p
                
    return flow, consumption, y, x
        

    
    
def show_vector( v ):
    return "[" + ", ".join( "{:0.2f}".format( x ) for x in v ) + "]"


def gradient_solver( f, x, debug_gradient = False ):
    flow, consumption, y = f.compute_flow( x, relaxed=True )
    beta = 0.4
    step = 0.1
    
    if False:
        for (n,attr) in flow.nodes( data=True ):
            print( n, attr['text'], attr['quantity'] )
            for (i,j,attr) in flow.edges( data=True ):
                print( i, j, attr['flow'] )

    for k in range( 1, 200  ):
        grad = f.gradient( x, y )
        if debug_gradient:
            print( "Gradient:    ", show_vector( grad ) )

        x_p = [ x_i + step * g_i 
                for x_i,g_i in zip( x, grad ) ]

        x_p = f.project_onto_simplices( x_p )
        f_p, c_p, y_p = f.compute_flow( x_p, relaxed = True )
        if debug_gradient:
            print( "y=", "{:0.5f}".format( y_p ),
                   "at", show_vector( x_p ) )

        # The "backtracking" algorithm compares real improvement with the
        # estimate step/2 * ||grad||^2.
        # If improvement is negative, or too small, decrease step size.
        #
        # But, because we project back into the allowed region, the effective
        # move we took could be much smaller.  Let's calculate what
        # the gradient "would have been" based on the move we actually made,
        # and use that lower value as our expected improvement.
        effective_grad = [ ( x1 - x2 ) / step for x1, x2 in zip( x_p, x ) ]
        if debug_gradient:
            print( "Eff grad:    ", show_vector( effective_grad ) )
        grad_squared = sum( x_i ** 2 for x_i in effective_grad )
        bound = (step / 2) * grad_squared
        if debug_gradient:
            print( "Improvement=", y_p - y, "bound=", bound )
        
        if y_p - y < bound:
            step *= beta
            if debug_gradient:
                print( "Reducing step to", step )
            if step < 0.000001:
                break

        if y_p > y:
            y = y_p
            x = x_p
            flow = f_p
            consumption = c_p

    return flow, consumption, y, x

def add_imports( g ):
    g.add_node( "IMP", tag="import", category="trade", text="import" )
    for n, t in g.nodes.data('category'):
        if t == 'source':
            continue
        if t == 'sink':
            continue
        if t == 'trade':
            continue
        if t == 'process':
            continue
        if t == 'factory':
            continue
        g.add_edge( "IMP", n )

    assert g.in_degree( "IMP" ) == 0
    return g

def add_tax( g ):
    edges = []
    g.add_node( "TAX", tag="tax", category="trade", text="tax" )
    for n, t in g.nodes.data('category'):
        if t == 'source':
            continue
        if t == 'sink':
            continue
        if t == 'trade':
            continue
        if t == 'process':
            continue
        if t == 'factory':
            continue
        g.add_edge( n, "TAX" )
        edges.append(  ( n, "TAX") )
    return edges

def flow_to_consumables( g,
                         metric = None,
                         all_initial_goods = 1000,
                         fixed_flows = [],
                         starting_flow = None,
                         verbose = True ):
    f = FlowModel( g, metric )
    f.find_parameters()
    if all_initial_goods > 0:
        f.set_initial_goods( all_initial_goods )
        
    for (i,j),a in fixed_flows:
        f.set_fixed_flow( i, j, a )
    if verbose:
        print( "Simplices:" )
        for s in f.simplices:
            print( "  ", s )

    if starting_flow is None:
        x_init = f.relaxed_start()
    else:
        x_init = []
        for s in f.simplices:
            total = 0.0
            xs = [ starting_flow.edges[e]['flow'] for e in s ]
            total = sum( xs )
            if total > 0.0:
                xs = [ x_i / total for x_i in xs ]
            x_init.extend( xs )
        
    flow, consumption, utility, x = gradient_solver( f, x_init )

    if verbose:
        print( "Gradient maximum:", x )
        print( "Utility =", utility )
        
    x_exact = f.unrelaxed_weights( x )

    if verbose:
        print( "Unrelaxed:", x_exact )
    flow, consumption, utility = f.compute_flow( x_exact, relaxed = False )

    if verbose:
        print( "Unrelaxed utility =", utility )
        for (n,attr) in flow.nodes( data=True ):
            print( n, attr['text'], attr['quantity'] )
        for n,v in consumption.items():
            print( n, v )

    flow, consumption, utility, x_final = augmenting_path_solver( f, x_exact )

    if verbose:
        print( "Optimized:", x_final )
        print( "Final utility =", utility )
        for (n,attr) in flow.nodes( data=True ):
            print( n, attr['text'], attr['quantity'] )
        for n,v in consumption.items():
            print( n, v )

    return flow, utility


    
