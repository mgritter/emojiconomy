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

def sigmoid_01( x ):
    return 2.0 * math.exp( 4 * x ) / ( math.exp( 4 * x ) + 1 ) - 1.0

def sigmoid_metric( consumables ):
    return sum( sigmoid_01( x ) for x in consumables.values() )


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
        # The model parameters are the weights attached to choice points,
        # where one good can be sent to any number of different destinations.
        #
        # We'll represent these as simplices which must sum to 1.
        for n in self.graph:
            if 'tag' not in self.graph.nodes[n]:
                continue
            
            if self.graph.nodes[n]['tag'] == 'source':
                for i,j in self.graph.out_edges( n ):
                    self.initial_goods.append( i )
            elif self.graph.nodes[n]['category'] == 'factory':
                self.equal_output.add( n )
            elif self.graph.nodes[n]['category'] == 'process':
                self.max_output.add( n )                
            elif self.graph.out_degree(n) > 1:
                simplex = list( self.graph.out_edges(n) )
                self.simplices.append( simplex )

        self.num_parameters = sum( len( s ) - 1 for s in self.simplices )

    def equal_distribution_parameters( self ):
        p = []
        for s in self.simplices:
            x = int( 100 / len(s)) / 100.0
            p.extend( [x] * ( len( s ) - 1 ) )
        return p

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
        
    def recenter_simplex( self, x ):
        x_prime = []
        pos = 0
        
        for simplex in self.simplices:    
            p_size = len( simplex ) - 1
            p = x[pos:pos+p_size]
            pos += p_size
            if sum( p ) > 1.0:
                tot = sum( p )
                p = [x_i / tot for x_i in p]
            x_prime.extend( p )
            
        return x_prime
                        
    def compute_flow( self, parameters, initial_production = None):
        if initial_production is None:
            initial_production = [1.0 for _ in self.initial_goods ]

        g = self.graph.copy()
        for n in g.nodes:
            g.nodes[n]['quantity'] = 0.0

        for i,j in g.edges:
            g[i][j]['weight'] = 1.0
            g[i][j]['flow'] = 0.0

        pos = 0
        for simplex in self.simplices:
            # Find the corresponding parameters
            p_size = len( simplex ) - 1
            p = parameters[pos:pos+p_size]
            pos += p_size
            
            # Handle overages gracefully
            if sum( p ) > 1.0:
                tot = sum( p )
                p = [x / tot for x in p]

            # Add last element of simplex
            p += [1.0 - sum(p)]
            
            for e,weight in zip( simplex, p ):
                g.edges[e]['weight'] = weight                
            
        for i,n in enumerate( self.initial_goods ):
            g.nodes[n]['quantity'] = initial_production[i]

        for n in topological_sort( g ):
            # If node is a multi-output processing plant, its outgoing
            # edges are tagged '=' and the input should be divided equally.
            if self.graph.nodes[n]['category'] == 'process':
                g.nodes[n]['quantity'] += \
                    sum( f for i,j,f in g.in_edges(n,data='flow') )
                outputs = g.out_edges( n )
                num_outputs = len( outputs )
                for o in outputs:
                    g.edges[o]['flow'] = g.nodes[n]['quantity'] / num_outputs
                    
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
                outputs = g.out_edges(n,data='weight')
                for (i,j,w) in outputs:
                    g.edges[(i,j)]['flow'] = g.nodes[n]['quantity'] * w
                    
        c = amount_consumed( g )
        return g, c, self.metric( c )

    def gradient( self, p, m = None, basis = None ):
        if m is None:
            _, _, m = self.compute_flow( p )

        grad = [ 0.0 for _ in range( len( p ) )  ]
        for i in range( len( p ) ):
            p_prime = list( p )
            delta = 0.001
            if p_prime[i] - delta < 0.0:
                delta = -0.001
            p_prime[i] += delta
            _, _, m_prime = self.compute_flow( p_prime )
            grad[i] = ( m_prime - m ) / delta
        return grad
    
def alternate_basis( n ):
    return [ [1] * i + [0] * (n-i) for i in range( 1, n+1 ) ]

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

def gradient_solver( f, x ):
    flow, consumption, y = f.compute_flow( x )
    while True:
        x = f.recenter_simplex( x )
        
        grad = f.gradient( x, y )
        if debug_gradient:
            print( "Gradient: ", grad )
            
        if sum( abs(g_i) for g_i in grad ) < 0.001:
            return flow, consumption, y, x

        # Remove directions where we can't move any further
        for i in range( len( x ) ):
            if x[i] == 1.0 and grad[i] > 0.0:
                grad[i] = 0.0
            elif x[i] == 0.0 and grad[i] < 0.0:
                grad[i] = 0.0

        # We'll only keep 0.01 precision in the parameters
        # so we would like to step enough to see at least one
        # parameter change.
        # i.e., | step * g_i | > 0.01
        min_step_size = min( smallest_step( x_i, g_i )
                             for x_i, g_i in zip( x, grad )
                             if g_i != 0.0  )
        
        # we can only go so far as 0 or 1
        max_step_size = min( largest_step( x_i, g_i )
                             for x_i, g_i in zip( x, grad )
                             if g_i != 0.0 )

        n_sizes = int( max_step_size / min_step_size )
        if debug_gradient:
            print( "Step size: ", min_step_size, "to", max_step_size,
                   "min/max =", n_sizes )

        # Very dumb search
        visited = []
        for n in range( 1, n_sizes + 1 ):
            step_size = min_step_size * n
            x_p = [ round( x_i + g_i * step_size )
                    for x_i,g_i in zip( x, grad ) ]
            f_p, c_p, y_p = f.compute_flow( x_p )
            visited.append( ( y_p, x_p, n, f_p, c_p ) )

        y_p, x_p, n, f_p, c_p = max( visited )
        if debug_gradient:
            print( n, "steps, best =", y_p, "at", x_p )
            
        if y_p < y:
            return flow, consumption, y, x
        else:
            y = y_p
            x = x_p
            flow = f_p
            consumption = c_p

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
    
    x_init = f.equal_distribution_parameters()
    flow, consumption, utility, x = gradient_solver( f, x_init )

    flow, consumption, utility, x = step_solver( f, x )
    
    print( "Final state:", x )
    for (n,attr) in flow.nodes( data=True ):
        print( n, attr['text'], attr['quantity'] )
    if False:
        for (i,j,attr) in flow.edges( data=True ):
            print( i, j, attr['flow'] )
    for n,v in consumption.items():
        print( n, v )
    print( "Utility =", utility )

    return flow


    
