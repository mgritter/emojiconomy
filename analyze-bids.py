class Seller(object):
    def __init__( self, line ):
        tokens = line.split( " " )
        self.iteration = int( tokens[0] )
        self.cost = float( tokens[1] )
        self.planet = tokens[2]
        # 3 type
        self.good = int( tokens[4] )
        self.amount = int( tokens[5] )
        # 6 available
        
        self.bids = []
        self.trade = None
        
class Bid(object):
    def __init__( self, line ):
        tokens = line.split( " " )
        self.iteration = int( tokens[0] )
        self.planet = tokens[1]
        # 2 type
        self.good = int( tokens[3] )
        self.amount = int( tokens[4] )

class Trade(object):
    def __init__( self, line ):
        tokens = line.split( " " )
        self.id = int( tokens[0] )
        self.iteration = int( tokens[1] )
        self.seller = tokens[2]
        self.s_good = int( tokens[3] )
        self.s_amount = int( tokens[4] )

        self.buyer = tokens[6]
        self.b_good = int( tokens[7] )
        self.b_amount = int( tokens[8] )

def load_logs( seller_log = "bids-auction.txt",
               bid_log = "bids-bidders.txt",
               trade_log = "bids-trade.txt" ):
    goods = set()
    sellers = []
    with open( seller_log, "r" ) as f:
        header = next( f )
        for line in f:
            sellers.append( Seller( line ) )


    with open( bid_log, "r" ) as f:
        header = next( f )
        for line in f:
            bid = Bid( line )
            sellers[bid.iteration].bids.append( bid )
            goods.add(bid.good)

    goods = list( sorted( goods ) )

    trades = []
    with open( trade_log, "r" ) as f:
        header = next( f )
        for line in f:
            trades.append( Trade( line ) )

    t = list( trades )
    for s in sellers:
        if len( t ) == 0:
            break
        if s.planet == t[0].seller and \
           s.good == t[0].s_good and \
           s.amount == t[0].s_amount:
            s.trade = t.pop( 0 )
    
    return sellers, goods, trades

def show_bids_by_contract( sellers, goods, trades,
                           show_multiple = True ):
    contracts = set()
    for s in sellers:
        contracts.add( (s.good, s.amount) )

    for (g, a) in sorted( contracts ):
        print( "----------------------------------------------" )
        print( "Good", g, "Amount", a )
        print( "----------------------------------------------" )
        timeseries = [s for s in sellers 
                      if s.good == g and s.amount == a]
        print( "Iteration",
               " ".join( "{:3}".format(g) for g in goods ) )

        for s in timeseries:
            line = "{:<9}".format( s.iteration )
            all_bids = { g : [] for g in goods }
            for b in s.bids:
                all_bids[b.good].append( b.amount )
            for g in goods:
                all_bids[g].sort()

            if s.trade is None:
                trade = ""
            else:
                trade = "[Trade {}]".format( s.trade.id )
                
            while True:
                best_bid = { g : all_bids[g].pop() for g in goods
                             if len( all_bids[g] ) > 0 }
                if len( best_bid ) == 0:
                    break
                for g in goods:
                    if g in best_bid:
                        line += " {:3}".format( best_bid[g] )
                    else:
                        line += "    "
                print( line, trade )
                if not show_multiple:
                    break
                trade = ""
                line = "{:<9}".format( "" )
            


if __name__ == "__main__":
    sellers, goods, trades = load_logs()
    show_bids_by_contract( sellers, goods, trades )
    
