from emojiconomy.grammar import run_econ, describe_econ
from soffit.display import drawSvg
from emojiconomy.flow import flow_to_consumables
import networkx as nx
import random
import math
import pickle

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

    def to_html( self, graph, out ):
        print( '<p>', file=out)
        for n,a in self.src_items.items():
            print( a, "x", graph.nodes[n]['tag'], file=out )
        print( " <=> ", file=out )
        for n,a in self.dst_items.items():
            print( a, "x", graph.nodes[n]['tag'], file=out )
        print( '</p>', file=out)

    def to_html_rev( self, graph, out ):
        print( '<p>', file=out)
        for n,a in self.dst_items.items():
            print( a, "x", graph.nodes[n]['tag'], file=out )
        print( " <=> ", file=out )
        for n,a in self.src_items.items():
            print( a, "x", graph.nodes[n]['tag'], file=out )
        print( '</p>', file=out)

        
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

def show_bids_asks( bids, asks ):
    array = {}
    cols = set()
    contract_sizes = set()
    for (i,i_a,e,e_a) in bids:
        if e not in array:
            array[e] = {}
        # "I will pay e_a of e for to import on i_a of i"
        array[e][i] = e_a
        cols.add( i )
        contract_sizes.add( i_a )
        
    rows = list( array.keys() )
    rows.sort()
    cols = list( cols )
    cols.sort()

    print( "Bids, contract sizes", " ".join( str(x) for x in contract_sizes ) )
    header = " ".join( [ "EXP\IMP" ] + [ "{:2}".format( c ) for c in cols ] )
    print( header )
    for exp in rows:
        row_items = [ "[{:5}]".format( exp ) ]
        for i in cols:
            if i in array[exp]:
                row_items.append( "{:2}".format( array[exp][i] ) )
            else:
                row_items.append( "  " )
        print( " ".join( row_items ) )


    array = {}
    cols = set()
    contract_sizes = set()
    for (e,e_a,i,i_a) in asks:
        if e not in array:
            array[e] = {}
        # "I will accept i_a of i to export e_a on e"
        array[e][i] = i_a
        cols.add( i )
        contract_sizes.add( e_a )
        
    rows = list( array.keys() )
    rows.sort()
    cols = list( cols )
    cols.sort()

    header = " ".join( [ "EXP\IMP" ] + [ "{:2}".format( c ) for c in cols ] )
    print()
    print( "Asks, contract sizes", " ".join( str(x) for x in contract_sizes ) )
    print( header )
    for exp in rows:
        row_items = [ "[{:5}]".format( exp ) ]
        for i in cols:
            if i in array[exp]:
                row_items.append( "{:2}".format( array[exp][i] ) )
            else:
                row_items.append( "  " )
        print( " ".join( row_items ) )


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

        # Cached bids, indexed by import
        self.cached_bids = {}
        self.cached_export_value = {}

    def num_available_for_export( self, n ):
        # Quantity represents flow into a node, including fixed flows.
        # But it doesn't subtract out how many were used for exports.
        input_amount = self.current_flow.nodes[n]['quantity']
        min_flow = self.current_flow.out_edges(n,data="min_flow")
        min_flow_required = sum( f for i,j,f in min_flow )
        return input_amount - min_flow_required 
        
    def has_items( self, n, min_amount = 1 ):
        if self.current_flow is None:
            return False
        return self.num_available_for_export( n ) >= min_amount
 
    def add_trade( self, t ):
        self.trades.append( t )

        # Invalidate bid cache
        self.cached_bids = {}
        self.cached_export_value = {}
        
        # Update current, should usually be the same as last "try"
        if self.last_trade == t.id:
            self.current_utility = self.last_utility
            self.current_flow = self.last_flow
        else:
            self.current_flow, self.current_utility = self.calc_utility()

    def export_value( self, good, amount = 1, recursive_depth = 0 ):
        if recursive_depth > 3:
            print( "Recursion too deep, giving up." )            
            return 0.0
        
        key = (good,amount)
        if key in self.cached_export_value:
            return self.cached_export_value[key]
        
        flow, utility = self.calc_utility( additional_exports=[(good,amount)] )
        if flow.graph['negative_flows']:
            return -10.0
        
        delta = utility - self.current_utility
        if delta > 0.0:
            self.recalc_utility( flow )
            # Try again
            return self.export_value( good, amount, recursive_depth + 1 )

        self.cached_export_value[key] = delta
        return delta

    def import_value( self, good, amount = 1 ):
        flow, utility = self.calc_utility( additional_imports=[(good,amount)] )
        delta = utility - self.current_utility
        return delta

    def minimum_profit( self, contract_size ):
        # 0.01 at 100
        # 0.0036 at 36
        # 0.0012 at 12
        # 0.0004 at 4
        return contract_size * 0.0001

    def bids_for_import( self, exportable_goods, import_good, contract_size ):
        """Given a good for purchase, what bids could pay for it?
        Return a truthful maximum price and we can run a 
        second-price auction."""
        key = ("bid",import_good,contract_size)
        if key in self.cached_bids:
            return self.cached_bids[key]

        min_profit = self.minimum_profit( contract_size )
        bids = []
        for g in exportable_goods:
            if g == import_good:
                continue
            # Find maximum amount, cap at 2x the contract size
            # Used to be 100 but I wanted to add larger contracts.
            lb = 0
            ub = min( contract_size * 2 + 1,
                      self.num_available_for_export(g) + 1 )
            lb_surplus = None
            while lb + 1 < ub:
                amount = (lb + ub) // 2
                flow, utility = self.calc_utility(
                    additional_exports=[(g,amount)],
                    additional_imports=[(import_good,contract_size)]
                )
                surplus = utility - self.current_utility
                #print( "Testing export of", amount, "=", surplus )
                # Need a positive surplus, 0 won't cut it.
                # Bumping lower bound to simulate overhead costs;
                # otherwise buyer surplus is very small.
                if surplus > min_profit:
                    lb_surplus = lb
                    lb = amount
                else:
                    ub = amount
            if lb_surplus is not None:
                #print( "Added bid at", lb )
                bids.append( (g,lb) )

        self.cached_bids[key] = bids
        return bids

    def asks_for_export( self, importable_goods, export_good, contract_size ):
        """Given a desired good, what goods would be accepted in exchange?
        i.e., "I need X lemons, what would I have to pay"?
        Return the minimum acceptable and we'll run a second-price 
        reverse auction."""

        if not self.has_items( export_good, contract_size ):
            return []

        key = ("ask",export_good,contract_size)
        if key in self.cached_bids:
            return self.cached_bids[key]
        
        min_profit = self.minimum_profit( contract_size )
        bids = []
        for g in importable_goods:
            if g == export_good:
                continue
            # Find minimum amount
            lb = 0
            lb_surplus = None
            ub = contract_size * 2 + 1
            while lb + 1 < ub:
                amount = (lb + ub) // 2
                flow, utility = self.calc_utility(
                    additional_exports=[(export_good,contract_size)],
                    additional_imports=[(g,amount)],
                )
                surplus = utility - self.current_utility
                if surplus > min_profit:
                    # Look for lower ask
                    ub = amount
                    ub_surplus = lb
                else:
                    # Look for higher ask
                    lb = amount

            if ub_surplus is not None:
                #print( "Added bid at", lb )
                bids.append( (g,ub) )

        self.cached_bids[key] = bids
        return bids
        
    def import_export_hints( self, tradable_goods ):
        suggested_imports = []
        suggested_exports = []
        for g in tradable_goods:
            i_value = self.import_value( g, amount=2 )
            if i_value > 0.0:
                suggested_imports.append( (i_value, g) )
            if self.has_items( g, 2 ):
                e_value = self.export_value( g, amount=2 )
                suggested_exports.append( (e_value, g) )
        suggested_imports.sort( reverse=True )
        suggested_exports.sort( reverse=True )
        return (suggested_imports, suggested_exports)
                
    def try_trade( self, t ):
        # FIXME: refactor to use additional_exports and additional_imports
        # instead of modifying the list of trades
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
                    

    def draw( self, prefix = "econ", flow = None ):
        if flow is None:
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

    def calc_utility( self,
                      additional_imports = [],
                      additional_exports = [] ):
        g2, ff = self.add_trades_to_graph(
            additional_imports = additional_imports,
            additional_exports = additional_exports
        )
        starting_flow = None
        if self.current_flow is not None:
            starting_flow = self.current_flow
        flow, utility = flow_to_consumables( g2,
                                             fixed_flows=ff,
                                             starting_flow=starting_flow,
                                             verbose=False )
        return flow, utility

    def recalc_utility( self, starting_flow ):
        """This is sort of a hack. Sometime exporting a good will kick
        the solver out of a local minimum and it'll find a better one.
        Verify that that's the case, and use that as the new baseline."""
        print( "Recalculating planet", self.id, "utility, current =",
               self.current_utility )
        g2, ff = self.add_trades_to_graph()
        flow, utility = flow_to_consumables( g2,
                                             starting_flow=starting_flow,
                                             fixed_flows=ff,
                                             verbose=False )
        print( "utility with new flow =", utility )
        if utility > self.current_utility:
            print( "Improvement found, adopting new baseline." )
            self.draw( prefix="suboptimal-before" )
            self.draw( prefix="suboptimal-after", flow=flow )
            self.current_flow = flow
            self.current_utility = utility
    
    def add_trades_to_graph( self, additional_exports = [],
                             additional_imports = [] ):
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

        for k,v in additional_exports:
            outgoing_flows[k] = outgoing_flows.get( k, 0 ) + v            
        for k,v in additional_imports:
            incoming_flows[k] = incoming_flows.get( k, 0 ) + v
                    
        flows =  []
        if len( incoming_flows ) > 0:
            flows += add_imports( g, incoming_flows )
        if len( outgoing_flows ) > 0:
            flows += add_exports( g, outgoing_flows )
        return g, flows

class Seller(object):
    def __init__( self ):
        self.planet = None
        self.delta = 0.0
        self.good = None
        self.amount = None
        self.auction_type = "sell"

class Buyer(object):
    def __init__( self ):
        self.planet = None
        self.good = None
        self.amount = None
    
class Auction(object):
    def __init__( self, galaxy ):
        self.galaxy = galaxy
        self.seller_log = None
        self.buyer_log = None
        
    def start_log( self, prefix="bids" ):
        self.seller_log = open( prefix + "-auction.txt", "w" )
        print( "iteration", "cost", "planet", "type",
               "good", "amount", "available",
               file=self.seller_log )

        self.buyer_log = open( prefix + "-bidders.txt", "w" )
        print( "iteration", "bidder", "type", "good", "amount",
               file=self.buyer_log )

    def stop_log( self ):
        if self.seller_log is not None:
            self.seller_log.close()
            self.seller_log = None
            
        if self.buyer_log is not None:
            self.buyer_log.close()
            self.buyer_log = None

    def choose_seller( self, iteration_count ):
        pass

    def choose_buyer( self, iteration_count, sellers, bids ):
        pass

    def successful_trade( self, trade ):
        pass

    def failed_trade( self, trade ):
        pass

    def failed_bidding( self, seller ):
        pass
    
    def collect_orders( self, iteration_count, seller ):
        """Collect the bids from all non-sellers."""

        # FIXME: handle asks_for_export too.
        
        bids = { g : [] for g in self.galaxy.useful_goods if g != seller.good }
    
        for p in self.galaxy.planets:
            if p == seller.planet:
                continue
            p_bids = self.galaxy.planets[p].bids_for_import(
                self.galaxy.useful_goods, seller.good, seller.amount )
            print( "Planet", p, "bids (good,amount) =", p_bids )
            for good, amount in p_bids:
                bids[good].append( (amount, p) )

        return bids

    def best_second_price( self, iteration_count, seller, bids ):
        """Second-price auction on each good; see which is most preferred
        by the seller."""

        seller_planet = self.galaxy.planets[seller.planet]
        
        preferred_bids = []
        for g, order_book in bids.items():
            order_book.sort()
            print( "Bids using good", g, " (amount,bidder) = ", order_book )

            if len( order_book ) == 0:
                continue
            elif len( order_book ) == 1:
                second_price, bidder = order_book[0]
            elif order_book[-1][0] != order_book[-2][0]:
                # Different prices
                second_price = order_book[-2][0]
                bidder = order_book[-1][1]
            else:
                # Tied bidders, pick one randomly.                
                second_price = order_book[-2][0]
                assert order_book[-1][0] == second_price
                high_bidders = [b for a,b in order_book if a == second_price ]
                print( "tied bidders:", high_bidders )                
                bidder = random.choice( high_bidders )


            _, utility = seller_planet.calc_utility(
                additional_exports=[(seller.good, seller.amount)],
                additional_imports=[(g,second_price)]
            )
            surplus = utility - seller_planet.current_utility
            print( "Second price:", second_price,
                   "bidder", bidder,
                   "exporter utility", utility,
                   "surplus", surplus )
            if utility > seller_planet.current_utility:
                preferred_bids.append( (utility, g, second_price, bidder ) )

        if len( preferred_bids ) == 0:
            print( "No acceptable bids." )
            return None
        
        preferred_bids.sort( reverse=True )
        b = Buyer()
        _, b.good, b.amount, b.planet = preferred_bids[0]
        return b

    def propose_trade( self, iteration_count ):
        # Identify a seller (or buyer) to lead this round
        s = self.choose_seller( iteration_count )
        if s is None:
            return None
        
        seller_available = self.galaxy.planets[s.planet].num_available_for_export( s.good )
        
        if self.seller_log is not None:
            print( iteration_count,
                   s.delta,
                   s.planet,
                   s.auction_type,
                   s.good,
                   s.amount,
                   seller_available,
                   file=self.seller_log, flush=True )

        book = self.collect_orders( iteration_count, s )
        if self.buyer_log is not None:
            for good, bids in book.items():
                for amount, p in bids:
                    print( iteration_count,
                           p, "buy", good, amount,
                           file=self.buyer_log )
        
        winner = self.choose_buyer( iteration_count, s, book )
        if winner is None:
            self.failed_bidding( s )
            return None

        t = Trade( self.galaxy.next_trade_id, s.planet, winner.planet  )
        self.galaxy.next_trade_id += 1
        t.src_items[s.good] = s.amount
        t.dst_items[winner.good] = winner.amount
        return t    
    
class CheapestFirstAuction(Auction):
    def __init__( self, galaxy ):
        super().__init__( galaxy )
        
        self.contract_schedule = [4, 12, 36, 100]
        self.contract_index = 3
        self.bid_exclusion = set()
        self.failed_count = 0

    def failed_trade( self, t ):
        for g in t.dst_items:
            s = Seller()
            s.planet = t.src
            s.good = g
            self.failed_bidding( s )
        
    def failed_bidding( self, seller ):
        # Implement a descending contract size
        self.bid_exclusion.add( (seller.planet, seller.good) )
        self.failed_count += 1
        if self.failed_count == 25:
            self.failed_count = 0
            if self.contract_index > 0:
                # Reset for next phase
                self.contract_index = self.contract_index - 1
                self.bid_exclusion = set()
            
                    
    def choose_buyer( self, i, s, b ):
        return self.best_second_price( i, s, b )
    
    def choose_seller( self, iteration_count ):
        galaxy = self.galaxy

        contract_size = self.contract_schedule[self.contract_index]
        print( "Contract size:", contract_size )
        
        # Find a low-value good to export, that has a use
        best_export = (-1.0, None, None)
        ties = []
        for contract_good in self.galaxy.useful_goods:
            export_values = []
            for p, planet in galaxy.planets.items():
                if not planet.has_items( contract_good, contract_size ):
                    continue
                # Delta should be <= 0.0, a 0.0 is the best
                delta = planet.export_value( contract_good, contract_size )
                print( "Planet", p, "good", contract_good, "delta", delta,
                       "available", planet.num_available_for_export( contract_good ) )
                if ( p, contract_good ) not in self.bid_exclusion:
                    export_values.append( (delta, p, contract_good) )

            # Exclude cases where all values are zero, not likely to work.
            if sum( 1 for d,_,_ in export_values if d < 0.0 ) == 0:
                continue
            
            best_export = max( [best_export] + export_values )
            # Keep track of who has an equally low bid so we can randomize
            # instead of deterministically picking the same planet all the time.
            ties = [ (v,p,g) for v,p,g in ties + export_values if
                     v == best_export[0] ]

        print( len( ties ), "best exports" )
        if len( ties ) == 0:
            # Reset at smaller contract size?
            self.bid_exclusion = {}
            self.failed_count = 0
            self.contract_index = min( self.contract_index - 1, 0 )
            return None        
        
        best_delta, best_exporter, best_good = random.choice( ties )
        print( "Chose exporter:", best_exporter,
               "good:", best_good,
               "delta:", best_delta )

        s = Seller()
        s.planet = best_exporter
        s.delta = best_delta
        s.good = best_good
        s.amount = contract_size
        s.auction_type = "sell"
        return s
        
class Galaxy(object):
    def __init__( self ):
        self.planets = {}
        self.goods = []
        self.next_trade_id = 100
        self.trades = {}
        self.trade_log = None
        self.buyer_log = None
        self.auction = CheapestFirstAuction(self)
        
    def start_log( self, prefix="bids" ):
        self.trade_log = open( prefix + "-trade.txt", "w" )
        print( "TradeID", "iteration",
               "exporter", "export_good", "export_amount", "export_surplus",
               "importer", "import_good", "import_amount", "import_surplus",
               file=self.trade_log )
        self.auction.start_log( prefix )

    def stop_log( self ):
        if self.trade_log is not None:
            self.trade_log.close()
            self.trade_log = None

        self.auction.stop_log()

    def draw_trade_report( self, prefix="planets" ):
        g = nx.Graph()
        for p in self.planets:
            g.add_node( p )

        for t in self.trades.values():
            a = min( t.src, t.dst )
            b = max( t.src, t.dst )
            edge = (a,b)
            
            if edge not in g.edges:
                g.add_edge( a, b, volume=0,
                            id="trade{}x{}".format( a, b ) )

            vol = sum( t.src_items.values() ) + sum( t.dst_items.values() )
            g.edges[edge]['volume'] += vol

        for e in g.edges():
            g.edges[e]['tag'] = str( g.edges[e]['volume'] )

        drawSvg( g, prefix + "-trade-volume.svg", program="neato",
                 dotFile = "debug-trade-volume.dot" )

        with open( "divs.html", "w" ) as out:
            for e in g.edges():
                (src,dst) = e
                a = min( src, dst )
                b = max( src, dst )
                print( '<div style="display: none;" id="detailstrade{}x{}">'.format( a, b ),
                       file=out )
                print( "<h2>Trade between", a, "and", b, "</h2>",
                       file=out )
                for t in self.trades.values():
                    if t.src == a and t.dst == b:
                        t.to_html( self.full_graph, out )
                    elif t.src == b and t.dst == a:
                        t.to_html_rev( self.full_graph, out )
                print( '</div>',
                       file=out )
            
        
        
    def draw_trade_graph( self, prefix="planets" ):
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

        self.useful_goods = [ g for g in self.goods if self.full_graph.out_degree( g ) != 0 ]

        for i in range( 1, 10 ):
            planet_id = i * 1111
            print( "Creating planet", planet_id )
            p = Planet( planet_id, self.full_graph )
            p.damage_graph()
            self.planets[planet_id] = p

            flow, utility = p.calc_utility()
            p.base_utility = utility
            p.current_utility = utility
            p.current_flow = flow

            if False:
                bids, asks = p.bids_asks( self.goods, 4 )
                show_bids_asks( bids, asks )                
                #print( "Contract size 4, bids" )
                #print( "IMP size   EXP upper_bound" )
                #for b in bids:
                #    print( b )
                #print( "Contract size 4, asks:" )
                #print( "EXP size   IMP lower_bound" )
                #for a in asks:
                #    print( a )

                            
    def create_auction_trade( self, iteration_count ):
        t = self.auction.propose_trade( iteration_count )
        if t is None:
            return False
        
        # FIXME: redundant for A?
        p_a = self.planets[t.src]
        p_b = self.planets[t.dst]
        surplus_a = p_a.try_trade( t )
        surplus_b = p_b.try_trade( t )

        if self.trade_log is not None:
            src_good, src_amount = list( t.src_items.items() )[0]
            dst_good, dst_amount = list( t.dst_items.items() )[0]
            
            print( t.id,
                   iteration_count,
                   t.src, src_good, src_amount, surplus_a,
                   t.dst, dst_good, dst_amount, surplus_b,
                   file=self.trade_log,
                   flush=True )

        if surplus_a > 0.0 and surplus_b > 0.0:
            self.trades[t.id] = t
            t.show( self.full_graph )
            p_a.add_trade( t )
            p_b.add_trade( t )
            
            self.auction.successful_trade( t )
            return True
        else:
            print( "Trade didn't work out, a=", surplus_a, "b=", surplus_b )
            self.auction.failed_trade( t )
            return False 
        
    def create_suggested_trade( self ):
        a, b = random.sample( self.planets.keys(), 2 )
        p_a = self.planets[a]
        p_b = self.planets[b]

        print( "Considering most desired trades between", a, "and", b )
        a_imp, a_exp = p_a.import_export_hints( self.goods )
        b_imp, b_exp = p_b.import_export_hints( self.goods )

        b_imp = dict( ( k,v) for v,k in b_imp )
        b_exp = dict( ( k,v) for v,k in b_exp )

        # Rank potential trades by total surplus
        potential = []
        for v_i,g_i in a_imp:
            assert v_i > 0.0
            # Is it on B's list as a possible export?
            if g_i not in b_exp:
                continue
            
            for v_e,g_e in a_exp:
                if g_e == g_i:
                    continue
                # Is it a possible import from B, at any price?
                if g_e not in b_imp:
                    continue

                quantity_export = 2
                quantity_import = 2
                a_value = v_i + v_e
                if a_value <= 0.0:
                    # x * v_i + v_e > 0.001
                    # x * v_i > (0.0001 -v_e)
                    # x > -v_e / v_i
                    quantity_import = int( math.ceil( ( 0.005 - v_e ) / v_i ) )
                    if quantity_import < 2 or quantity_import > 20:
                        break
                    a_value = v_i * quantity_import / 2.0 + v_e
                
                b_value = b_imp[ g_e ] * quantity_import / 2.0 + b_exp[ g_i ]
                if b_value <= 0.0:
                    # Not interested
                    continue

                surplus = a_value + b_value
                potential.append( (surplus,
                                   (g_i, quantity_import),
                                   (g_e, quantity_export) ) )

            potential.sort( reverse=True )

        for expected, (g_i, q_i), (g_e, q_e) in potential:
            t = Trade( self.next_trade_id, a, b )
            t.src_items[g_e] = q_e
            t.dst_items[g_i] = q_i
            surplus_a = p_a.try_trade( t )
            surplus_b = p_b.try_trade( t )
            print( "Expected", expected, "a", surplus_a, "b", surplus_b,
                   "total", surplus_a + surplus_b, "for", q_i, "<->", q_e )
            if surplus_a > 0.0 and surplus_b > 0.0:
                self.next_trade_id += 1
                t.show( self.full_graph )
                p_a.add_trade( t )
                p_b.add_trade( t )
                self.trades[t.id] = t
                return True

        return False
        
    def create_random_trade( self, force = False ):
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

    galaxy.start_log()

    for n in range( 200 ):
        if galaxy.create_auction_trade( n ):
            galaxy.report()
        #if galaxy.create_suggested_trade():
        #    galaxy.report()
        #if galaxy.create_trade():
        #    galaxy.report()

        if n % 33 == 32:
            galaxy.draw( prefix = "step-{}".format( n ),
                         draw_planets = True )
            
    galaxy.stop_log()
    with open( "galaxy.pickle", "wb" ) as o:
        pickle.dump( galaxy, o )

    galaxy.draw( draw_planets = True )
    galaxy.draw_trade_graph()

import sys

galaxy = None

if __name__ == "__main__":
    if len( sys.argv ) > 1 and sys.argv[1] == 'reload':
        with open( "galaxy.pickle", "rb") as f:
            galaxy = pickle.load( f )
        if len( sys.argv ) > 2 and sys.argv[2] == 'auction':
            galaxy.auction = CheapestFirstAuction( galaxy )
            galaxy.create_auction_trade( 300 )
    else:
        main()
