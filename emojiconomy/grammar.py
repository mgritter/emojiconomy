import soffit.application as soffit
from soffit.display import drawSvg
import os

from emojiconomy.emoji import emojify_graph
from emojiconomy.flow import annotate_source_edges, flow_to_consumables

#import networkx as nx

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

econ_fn = os.path.join(__location__, 'econ.soffit.json')
econ_grammar = soffit.loadGrammar( econ_fn )

types = [
    "plant",
    "plant_proc",
    "plant_prod",
    "food",
#    "ore",
#    "ore_proc",
#    "machine",
#    "house",
#    "toy"
]

def remove_unused( g ):
    for n, tag in list( g.nodes.data( 'tag' ) ):
        if "allowed_" in tag:
            g.remove_node( n )

def pretty_print_list( nodes ):
    if len( nodes ) == 1:
        return nodes[0]['text']
    elif len( nodes ) == 2:
        return " and ".join( n['text'] for n in nodes)
    else:
        txt = ", ".join( n['text'] for n in nodes[:-1] )
        return txt + ", and " + nodes[-1]['text']
        
def describe_econ( g ):
    for n in g.nodes:
        category = g.nodes[n]['category']
        if category == "process":
            inputs  = list( g.nodes[p] for p in g.predecessors( n ) )
            outputs  = list( g.nodes[s] for s in g.successors( n ) )
            print( g.nodes[n]['text'], "processes",
                   pretty_print_list( inputs ), "to",
                   pretty_print_list( outputs ) )
        elif category == "factory":
            inputs  = list( g.nodes[p] for p in g.predecessors( n ) )
            outputs  = list( g.nodes[s] for s in g.successors( n ) )
            print( g.nodes[n]['text'], "combines",
                   pretty_print_list( inputs ), "to make",
                   pretty_print_list( outputs ) )
        elif category == "sink":
            inputs  = list( g.nodes[p] for p in g.predecessors( n ) )
            print( "Population consumes",
                   pretty_print_list( inputs ) )
        elif category == "source":
            outputs  = list( g.nodes[s] for s in g.successors( n ) )
            print( "Planet grows",
                   pretty_print_list( outputs ) )
            

def run_econ( default_maximum = 10, maximums = {} ):
    # Augment the start grammar with maximum number of goods per type.
    init_graph = econ_grammar.start.copy()

    nodeNumber = 0
    for t in types:        
        m = maximums.get( t, default_maximum )
        for _ in range( m ):
            init_graph.add_node( "M" + str( nodeNumber ), tag="allowed_"+ t )
            nodeNumber += 1
                            
    a = soffit.ApplicationState(
        initialGraph = init_graph,
        grammar = econ_grammar
    )

    a.run( 20 )
    remove_unused( a.graph )
    emojify_graph( a.graph )

    describe_econ( a.graph )
    
    drawSvg( a.graph, "econ.svg" )

    g2 = a.graph.copy()
    annotate_source_edges( g2 )
    flow_to_consumables( g2, None )
    

if __name__ == "__main__":
    run_econ(3)
    
