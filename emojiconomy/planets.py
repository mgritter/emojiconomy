from emojiconomy.grammar import run_econ, describe_econ
from soffit.display import drawSvg
from emojiconomy.flow import flow_to_consumables

def main():
    full_graph = run_econ( 3 )

    describe_econ( full_graph )
    drawSvg( full_graph, "econ-full.svg" )
    
    flow, utility = flow_to_consumables( full_graph,
                                         verbose=False )
    for e in flow.edges:
        flow.edges[e]['tag'] = str( int( flow.edges[e]['flow'] ) )

    print( "Full economy utility:", utility )
    drawSvg( flow, "econ-full-flow.svg" )

    source = None
    for n, category in full_graph.nodes( data="category" ):
        if category == "source":
            source = n
            continue
        if category == "sink":
            continue
        if category == "process" or category == "factory":
            g2 = full_graph.copy()
            g2.remove_node( n )
            print( "Destroying", full_graph.nodes[n]['text'] )
            flow, utility = flow_to_consumables( g2,
                                                 verbose=False )
            print( "Utility =", utility )
        if category == "plant" and source is not None:
            g2 = full_graph.copy()
            g2.remove_edge( source, n )
            print( "Destroying", full_graph.nodes[n]['text'] )
            flow, utility = flow_to_consumables( g2,
                                                 verbose=False )
            print( "Utility =", utility )
            
            
            
            
        
            

if __name__ == "__main__":
    main()
