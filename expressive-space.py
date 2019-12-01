import emojiconomy.grammar as eg
import emojiconomy.flow as flow
import soffit.display as sd
from networkx.drawing.nx_agraph import to_agraph
import matplotlib.pyplot as plt
import numpy as np
import io

width = 10
height = 10

num_examples = width * height

def toSvg( g ):
    toDraw = sd.tagsToLabels( g )
    try:
        del toDraw.graph['join']
        del toDraw.graph['rename']
    except KeyError:
        pass

    aGraph = to_agraph( toDraw )
    aGraph.graph_attr['overlap'] = 'false'
    aGraph.graph_attr['outputorder'] = 'edgesfirst'
    size = 150 / 96.0
    aGraph.graph_attr['size'] = "{},{}!".format( size, size )
    aGraph.node_attr['style']='filled'
    aGraph.node_attr['fillcolor']='white'
    
    return aGraph.draw( format="svg", prog="dot" )

htmlHeader="""<!DOCTYPE html>
<html lang="en">
<head>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8"/>
</head>
<body>
"""

summarySection="""
<h2>Emojiconomy stats</h2>
<div>
{}
</div>
"""

detailSection="""
<h2>Emojiconomy graphs indexed by (degrees of freedom, consumables, intermediate goods)</h2>
<div style="display: grid; grid-template-columns: {}; grid-column-gap: 2px;">
"""

out = open( "display.html", "w" )

print( htmlHeader, file=out )

#print( "NODES", "DF", "CONSUMABLES", "INTERMEDIATE" )

examples = []

class Example(object):
    def __init__( self ):
        pass
    
    def key( self ):
        return (self.num_df, self.num_consumables, self.num_intermediate)

    def coords( self ):
        # X, Y
        return (self.num_df, self.num_consumables )

max_coords = (0,0)

for n in range(num_examples):
    g = eg.run_econ(3,verbose=False)
    svg = toSvg( g ).decode( "utf-8" )

    f = flow.FlowModel( g )
    f.find_parameters()
    num_simplices = len( f.simplices )
    if False:
        for s in f.simplices:
            for i,j in s:
                print( "{} {} -> {} {}".format(
                    i, g.nodes[i]['category'],
                    j, g.nodes[j]['category'] ))
    num_df = f.num_parameters - num_simplices

    sinks = [n for n,t in g.nodes.data('tag') if t == 'sink']
    num_consumables = sum( g.in_degree( s ) for s in sinks )
    num_plants = sum( 1 for _,t in g.nodes.data('category') if t == 'plant' )
    num_proc = sum( 1 for _,t in g.nodes.data('category') if t == 'plant_proc' )
    num_prod = sum( 1 for _,t in g.nodes.data('category') if t == 'plant_prod' )
    num_food = sum( 1 for _,t in g.nodes.data('category') if t == 'food' )

    num_intermediate = num_plants + num_proc + num_prod + num_food - num_consumables

    x = Example()
    x.g = g
    x.svg = svg
    x.num_nodes = len( g.nodes )
    x.num_df = num_df
    x.num_consumables = num_consumables
    x.num_intermediate = num_intermediate
    examples.append( x )

    (i,j) = x.coords()
    max_coords = ( max( i + 1, max_coords[0] ),
                   max( j + 1, max_coords[1] ) )

#
# HEATMAP
#

shape = ( max_coords[1], max_coords[0] ) # Y, X
frequency = np.zeros( shape=shape )

for x in examples:
    (i,j) = x.coords()
    frequency[j,i] += 1

fig, ax = plt.subplots()
im = ax.imshow(frequency)

ax.set_xticks(np.arange(max_coords[0]))
ax.set_xlabel("degrees of freedom")
ax.set_yticks(np.arange(max_coords[1]))
ax.set_ylabel("number of consumables")

for i in range(max_coords[0]):
    for j in range(max_coords[1]):
        text = ax.text(i, j,
                       str(frequency[j,i]),
                       ha="center", va="center", color="w")

buf = io.StringIO()
plt.savefig( buf, format="svg")

print( summarySection.format( buf.getvalue() ), file=out )

#
# DETAILED EXAMPLES
#
examples.sort( key= lambda x : x.key() )

last_key = None

print( detailSection.format( " ".join( ["150px"] * width ) ), file=out )

next_id = 1
for x in examples:
    if x.key() == last_key:
        print( "<div id=\"d{}\"><p>&nbsp;</p>{}</div>".format(
            next_id,
            sd.removeHeader( x.svg ) ), file=out )
    else:
        print( "<div id=\"d{}\"><p>{}</p>\n{}</div>".format(
            next_id,
            str( x.key() ),
            sd.removeHeader( x.svg ) ), file=out )
        last_key = x.key()
    next_id += 1

htmlFooter="""
<script type="text/javascript">
var last_div = null;
var last_svg = null;

function select_div() {
  if ( last_div != null ) {
     last_div.style.gridColumnEnd = "span 1";
     last_div.style.gridRowEnd = "span 1";
     last_svg.style.transform = "scale(1,1)";
  }
  if ( last_div == this ) {
     last_div = null;
  } else {
     last_div = this;
     this.style.gridColumnEnd = "span 6";
     this.style.gridRowEnd = "span 6";
     for (c of this.children) {
        if (c.tagName == "svg") {
            last_svg = c;           
            c.style.transform = "scale(6,6)";
            c.style.transformOrigin = "0 0";
        }
     }
  }
}

for (var i = 1; i <= 100; ++i) {
  var name = "d" + i;
  var el = document.getElementById(name);
  if ( el != null ) {
    el.addEventListener('click', select_div, false);
  }
}
</script>
</div>
</body>
</html>
"""


print( htmlFooter, file=out )

out.close()
