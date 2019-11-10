import soffit.application as soffit
from soffit.display import drawSvg
import os

from emojiconomy.emoji import emojify_graph

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
    
    drawSvg( a.graph, "econ.svg" )

if __name__ == "__main__":
    run_econ(3)
    
