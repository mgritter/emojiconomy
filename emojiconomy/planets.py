from emojiconomy.grammar import run_econ, describe_econ
from soffit.display import drawSvg
from emojiconomy.flow import flow_to_consumables
import random

class Trade(object):
    def __init__( self, id, src, dst ):
        self.id = id
        self.src = src    
        self.dst = dst
        self.src_items = {}  # exports from src 
        self.dst_items = {}  # exports from dst
        
    def random_items( self, goods ):
        a, b= random.sample( goods, 2 )
        a_amount = random.randint( 2, 10 ) 
        b_amount = random.randint( 2, 10 ) 
        self.src_items[a] = self.src_items.get( a, 0 ) + a_amount 
        self.dst_items[b] = self.dst_items.get( b, 0 ) + b_amount 

def add_imports( g, amounts ):
    flows = []
    g.add_node( "IMP", tag="📦", category="trade", text="import" )
    for n, a in amounts.items():
        g.add_edge( "IMP", n )
        flows.append( ( ("IMP", n), a ) )
    return flows
        
def add_exports( g, amounts ):
    flows = []
    g.add_node( "EXP", tag="🚀", category="trade", text="export" )
    for n, a in amounts.items():
        g.add_edge( n, "EXP" )
        flows.append( ( (n, "EXP"), a ) )
    return flows


class Planet(object):
    def __init__( self, id, graph ):
        self.id = id
        self.graph = graph.copy()
        self.trades = []
        self.base_utility = 0.0
        
    def damage_graph( self ):
        sources = [ n for n,t in self.graph.nodes.data( 'tag' )
                    if t == 'source']
        source = sources[0]
        goods = [ g for g in goods_iter( self.graph )
                  if (source,g) in self.graph.edges ]
        producers = list( producers_iter( self.graph ) )
        
        damage = random.sample( goods + producers, k=1 )[0]
        if damage in goods:
            self.graph.remove_edge( source, damage )
        else:
            self.graph.remove_node( damage )

    def calc_utility( self ):
        g2, ff = self.add_trades_to_graph()
        flow, utility = flow_to_consumables( g2,
                                             fixed_flows=ff,
                                             verbose=False )
        return flow, utility
        
    def add_trades_to_graph( self ):
        g = self.graph.copy()
        incoming_flows = {}
        outgoing_flows = {}
        for t in self.trades:
            if t.src == self.id:
                my_exports = t.src_items
                my_imports = t.dst_items
            else:
                my_exports = t.dst_items
                my_imports = t.src_items
                
            for k, v in my_exports.items():
                outgoing_flows[k] = outgoing_flows.get( k, 0 ) + v
            for k, v in my_imports.items():
                incoming_flows[k] = incoming_flows.get( k, 0 ) + v

        flows =  []
        if len( incoming_flows ) > 0:
            flows += add_imports( g, incoming_flows )
        if len( outgoing_flows ) > 0:
            flows += add_exports( g, outgoing_flows )
        return g, flows
        
def goods_iter( g ):
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
        yield n

def producers_iter( g ):
    for n, t in g.nodes.data('category'):
        if t == 'factory':
            yield n
        if t == 'process':
            yield n

def draw_flow( flow, fn ):
    for e in flow.edges:
        flow.edges[e]['tag'] = str( int( flow.edges[e]['flow'] ) )
    drawSvg( flow, fn, program="dot" )

def explore_trade( graph, goods, base_value ):
    graph = graph.copy()
    
    imp, exp = random.sample( goods, 2 )
    fixed = add_imports( graph, { imp :  random.randint( 2, 10 ) } ) \
            + add_exports( graph, { exp :  random.randint( 2, 10 ) } )

    print( fixed )

    flow, utility = flow_to_consumables( graph,
                                         fixed_flows=fixed,
                                         verbose=False )
    print( "Trade utility: ", utility - base_value )
    return flow, utility
    
        
def main():
    full_graph = run_econ( 3 )

    describe_econ( full_graph )
    drawSvg( full_graph, "econ-full.svg", program="dot" )
    
    flow, utility = flow_to_consumables( full_graph,
                                         verbose=False )
    print( "Full economy utility:", utility )
    draw_flow( flow, "econ-full-flow.svg" )

    planet = Planet( 1, full_graph )
    print( "Damaging planet" )
    planet.damage_graph()
    flow, utility = planet.calc_utility()
    print( "Reduced utility:", utility )

    goods = list( goods_iter( full_graph ) )
    for k in range( 5 ):
        t = Trade( 0xff, 1, 2 )
        t.random_items( goods )
        
        planet.trades.append( t )
        f2, u2 = planet.calc_utility()
        print( "Considered trade:", u2 )
        if u2 <= utility:
            planet.trades.pop()
        else:
            utility = u2
            flow = f2
    
    draw_flow( flow, "econ-{}.svg".format( planet.id ) )
            

if __name__ == "__main__":
    main()
