from emojiconomy.grammar import run_econ, describe_econ
from soffit.display import drawSvg
from emojiconomy.flow import flow_to_consumables
import networkx as nx
import random

class Trade(object):
    def __init__( self, id, src, dst ):
        self.id = id
        self.src = src    
        self.dst = dst
        self.src_items = {}  # exports from src 
        self.dst_items = {}  # exports from dst
        
    def random_items( self, goods ):
        a, b = random.sample( goods, 2 )
        a_amount = random.randint( 2, 10 ) 
        b_amount = random.randint( 2, 10 ) 
        self.src_items[a] = self.src_items.get( a, 0 ) + a_amount 
        self.dst_items[b] = self.dst_items.get( b, 0 ) + b_amount 

    def show( self, graph ):
        print( "Trade ID", self.id )
        for n,a in self.src_items.items():
            print( self.src, "->", a, "x", graph.nodes[n]['text'], n )
        for n,a in self.dst_items.items():
            print( self.dst, "<-", a, "x", graph.nodes[n]['text'], n )

    def increase_dst_count( self, amount=1 ):
        for n in self.dst_items:
            self.dst_items[n] += amount
        
    def increase_src_count( self, amount=1 ):
        for n in self.src_items:
            self.src_items[n] += amount
                          
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
        self.current_utility = 0.0
        self.current_flow = None
        
        # Cached in try_trade for use in add_trade
        self.last_trade = None
        self.last_flow = None
        self.last_utility = None
        

    def has_items( self, n, min_amount = 1 ):
        if self.current_flow is None:
            return False
        return self.current_flow.nodes[n]['quantity'] >= min_amount
        
    def add_trade( self, t ):
        self.trades.append( t )
        if self.last_trade == t.id:
            self.current_utility = self.last_utility
            self.current_flow = self.last_flow
        else:
            self.current_flow, self.current_utility = self.calc_utility()
        
    def try_trade( self, t ):
        self.trades.append( t )
        flow, utility = self.calc_utility()
        self.trades.pop()
        
        self.last_trade = t.id
        self.last_utility = utility
        self.last_flow = flow
        
        if flow.graph['negative_flows']:
            return -1.0
        else:
            return utility - self.current_utility
        
    def report( self ):
        print( "Planet {:2d} utility {:6f} surplus {:6f} trades {}".format
               ( self.id,
                 self.current_utility,
                 self.current_utility - self.base_utility,
                 len( self.trades ) ) )
                    

    def draw( self, prefix = "econ" ):
        flow, utility = self.calc_utility()
        fn = "{}-{}-flow.svg".format( prefix, self.id )
        draw_flow( flow, fn )
                    
    def damage_graph( self ):
        sources = [ n for n,t in self.graph.nodes.data( 'tag' )
                    if t == 'source']
        source = sources[0]
        goods = [ g for g in goods_iter( self.graph )
                  if (source,g) in self.graph.edges ]
        producers = list( producers_iter( self.graph ) )
        
        damage = random.sample( goods + producers, k=1 )[0]
        print( "Removing", damage )
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

class Galaxy(object):
    def __init__( self ):
        self.planets = {}
        self.goods = []
        self.next_trade_id = 100
        self.trades = {}
        
    def draw_trade_graph( self, prefix="planets-" ):
        g = nx.DiGraph()
        for p in self.planets:
            g.add_node( p )

        for t in self.trades.values():
            edge = (t.src, t.dst)
            rev_edge = (t.dst, t.src)
            
            if edge not in g.edges:
                g.add_edge( t.src, t.dst, items={} )
                g.add_edge( t.dst, t.src, items={} )
                
            for k, v in t.src_items.items():
                g.edges[edge]['items'][k] = \
                    g.edges[edge]['items'].get( k, 0 ) + v
                
            for k, v in t.dst_items.items():
                g.edges[rev_edge]['items'][k] = \
                    g.edges[rev_edge]['items'].get( k, 0 ) + v

        before = list( g.edges.data( 'items' ) )

        nextId = 1000        
        for i, j, items in before:
            description = []
            for k,v in items.items():
                emoji = self.full_graph.nodes[k]['tag']
                description.append( "{} {}".format( v, emoji ) )

            g.add_node( nextId,
                        tag = " ".join( description ),
                        shape = "rectangle" )

            g.add_edge( i, nextId, arrowhead="none" )
            g.add_edge( nextId, j )
            g.remove_edge( i, j )
            nextId += 1

        drawSvg( g, prefix + "-trade.svg", program="neato",
                 dotFile = "debug-trade.dot" )
        
    def draw( self, prefix="econ", draw_planets = False ):
        describe_econ( self.full_graph )
        drawSvg( self.full_graph, prefix + "-full.svg", program="dot" )
    
        flow, utility = flow_to_consumables( self.full_graph,
                                             verbose=False )
        print( "Full economy utility:", utility )
        draw_flow( flow, prefix + "-full-flow.svg" )

        if draw_planets:
            for id, p in self.planets.items():
                p.draw( prefix=prefix )
            
    def report( self ):
        print()
        for id, p in self.planets.items():
            p.report()
        
    def create( self, num_planets = 10, max_good_types = 3  ):
        self.full_graph = run_econ( max_good_types )
        self.goods = list( goods_iter( self.full_graph ) )
        
        for planet_id in range( 1, 10 ):
            print( "Creating planet", planet_id )
            p = Planet( planet_id, self.full_graph )
            p.damage_graph()
            self.planets[planet_id] = p

            flow, utility = p.calc_utility()
            p.base_utility = utility
            p.current_utility = utility
            p.current_flow = flow

    def create_trade( self, force = False ):
        a, b = random.sample( self.planets.keys(), 2 )
        p_a = self.planets[a]
        p_b = self.planets[b]

        while True:
            t = Trade( self.next_trade_id, a, b )
            t.random_items( self.goods )

            # Try again if no quantity of the chosen goods exists.
            available = min( p_a.has_items( n, q )
                             for n,q in t.src_items.items() )
            if not available:
                continue
            available = min( p_b.has_items( n, q )
                             for n,q in t.dst_items.items() )
            if not available:
                continue
            
            break
          
        self.next_trade_id += 1    

        last_surplus_a = None
        last_surplus_b = None
        
        while True:
            surplus_a = p_a.try_trade( t )
            surplus_b = p_b.try_trade( t )

            if force or ( surplus_a > 0 and surplus_b > 0 ):
                print( "New trade:", surplus_a, surplus_b )
                t.show( self.full_graph )
                p_a.add_trade( t )
                p_b.add_trade( t )
                self.trades[t.id] = t
                return True
        
            print( "Rejected trade:",
                   surplus_a, surplus_b,
                   t.src_items, t.dst_items )
            
            if surplus_a <= 0 and surplus_b <= 0:
                return False

            if surplus_a > 0:
                amount = 1
                # Check for progress
                if last_surplus_b != None:
                    if surplus_b <= last_surplus_b:
                        return False
                    amount = -surplus_b / ( surplus_b - last_surplus_b )
                    print( "Estimate to positive:", amount )
                    amount = min( int( amount ), 10 )
                    amount = max( amount, 1 )
                    
                last_surplus_b = surplus_b
                last_surplus_a = None
                print( "Incrementing src", amount, "and retrying")
                t.increase_src_count( amount )

            elif surplus_b > 0:
                amount = 1
                # Check for progress
                if last_surplus_a != None:
                    if surplus_a <= last_surplus_a:
                        return False
                    amount = -surplus_a / ( surplus_a - last_surplus_a )
                    print( "Estimate to positive:", amount )
                    amount = min( int( amount ), 10 )
                    amount = max( amount, 1 )
                    
                last_surplus_a = surplus_a
                last_surplus_b = None
                print( "Incrementing dst", amount, "and retrying")
                t.increase_dst_count( amount )



        


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

        
def main():
    galaxy = Galaxy()
    galaxy.create()

    galaxy.draw()
    galaxy.report()

    for n in range( 200 ):
        if galaxy.create_trade():
            galaxy.report()

    galaxy.draw( draw_planets = True )
    galaxy.draw_trade_graph()
        

if __name__ == "__main__":
    main()
