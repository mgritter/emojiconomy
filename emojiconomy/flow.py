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

        self.equal_output = set()
        self.max_output = set()

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
                self.equal_output.add( n )
            elif self.graph.nodes[n]['category'] == 'process':
                self.max_output.add( n )                
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

    # FIXME
    def in_range( self, x ):
        if len( x ) == 0:
            return True
        if max( x_i <= 0 or x_i >= 1.0 for x_i in x ):
            return False

        pos = 0
        for simplex in self.simplices:    
            p_size = len( simplex ) - 1
            p = x[pos:pos+p_size]
            pos += p_size

            if sum( p ) > 1.0:
                return False
            
        return True
        
    def project_onto_simplices( self, x_0 ):
        """Project a relaxed solution onto its nearest L2-norm neighbor
        that obeys the simplex constraints.
        """
        z = []
        for simplex, x in self.simplex_iter( x_0 ):
            z.extend( project_onto_standard_simplex( x ) )
            
        return z
                        
    def compute_flow( self, parameters, initial_production = None,
                      relaxed = False ):
        if initial_production is None:
            initial_production = [1000 for _ in self.initial_goods ]

        g = self.graph.copy()
        for n in g.nodes:
            g.nodes[n]['quantity'] = 0

        for i,j in g.edges:
            g[i][j]['flow'] = 0
            if relaxed:
                g[i][j]['weight'] = 1.0
            else:
                # Infinity!
                g[i][j]['weight'] = 1000000
            
        pos = 0
        for simplex, x in self.simplex_iter( parameters ):
            for e,weight in zip( simplex, x ):
                g.edges[e]['weight'] = weight

        for i,n in enumerate( self.initial_goods ):
            g.nodes[n]['quantity'] = initial_production[i]

        # FIXME: draw flows from source
        
        for n in topological_sort( g ):
            # If node is a multi-output processing plant, its outgoing
            # edges are tagged '=' and the input should be divided equally.
            if self.graph.nodes[n]['category'] == 'process':
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
                g.nodes[n]['quantity'] += \
                    sum( f for i, j, f in g.in_edges(n,data='flow') )
                available = g.nodes[n]['quantity'] 
                outputs = g.out_edges(n,data='weight')
                if relaxed:
                    # Interpret as 0--1 simplex
                    for (i,j,w) in outputs:
                        g.edges[(i,j)]['flow'] = available * w
                else:
                    # Interpret as item counts, if feasible
                    total = sum( w for i,j,w in outputs )
                    if total < available:
                        for (i,j,w) in outputs:
                            g.edges[(i,j)]['flow'] = w
                    else:
                        # Infeasible, scale down
                        for (i,j,w) in outputs:
                            rounded_flow = available * w // total
                            g.edges[(i,j)]['flow'] = rounded_flow
                            
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
    
    def unrelaxed_weights( self, x, initial_production = None ):
        flows, _, m = self.compute_flow( x,
                                         initial_production=initial_production,
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

debug_gradient = True
debug_step = True

def pairwise_sum( x, delta ):
    return [ round(x_i + delta_i) for x_i, delta_i in zip( x, delta ) ]

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
    
def step_solver( f, x ):
    x = f.recenter_simplex( x )
    flow, consumption, y = f.compute_flow( x )
    step = 0.01
    
    print( "Start:", y, x )

    visited = set()
    visited.add( tuple(x) )
    
    while True:
        allSteps = itertools.product( [-step, 0.0, step], repeat=len( x ) )
        next_xs = [ pairwise_sum( x, d ) for d in allSteps ]
        
        neighborhood = [ (x_p, f.compute_flow( x_p )) for x_p in next_xs
                         if f.in_range( x_p ) and not tuple(x_p) in visited ]

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

def show_vector( v ):
    return "[" + ", ".join( "{:0.2f}".format( x ) for x in v ) + "]"


def gradient_solver( f, x ):
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

        # If improvement is negative, or too small, decrease step size
        grad_squared = sum( x_i ** 2 for x_i in grad )
        bound = (step / 2) * grad_squared
        print( "Improvement=", y_p - y, "bound=", bound )
        if y_p - y < bound:
            step *= beta
            print( "Reducing step to", step )
            if step < 0.00001:
                break

        if y_p > y:
            y = y_p
            x = x_p
            flow = f_p
            consumption = c_p

    return flow, consumption, y, x

def add_imports( g ):
    g = g.copy()
    g.add_node( "T", tag="import", category="trade", text="import" )
    for n, t in g.nodes.data('category'):
        if t == 'unassigned':
            continue
        if t == 'process':
            continue
        if t == 'factory':
            continue
        if n == "T":
            continue
        g.add_edge( "T", n )

    assert g.in_degree( "T" ) == 0
    return g
    
def flow_to_consumables( g, metric ):
    f = FlowModel( g, metric )
    f.find_parameters()
    print( f.simplices )
    
    x_init = f.relaxed_start()
    flow, consumption, utility, x = gradient_solver( f, x_init )

    # flow, consumption, utility, x = step_solver( f, x )
    
    print( "Gradient maximum:", x )
    print( "Utility =", utility )
    if False:
        for (n,attr) in flow.nodes( data=True ):
            print( n, attr['text'], attr['quantity'] )
    if False:
        for (i,j,attr) in flow.edges( data=True ):
            print( i, j, attr['flow'] )
    if False:
        for n,v in consumption.items():
            print( n, v )

    x_exact = f.unrelaxed_weights( x )
    print( "Unrelaxed:", x_exact )
    flow, consumption, utility = f.compute_flow( x_exact, relaxed = False )
    print( "Unrelaxed utility =", utility )
    for (n,attr) in flow.nodes( data=True ):
        print( n, attr['text'], attr['quantity'] )
    for n,v in consumption.items():
        print( n, v )
    
    return flow


    
