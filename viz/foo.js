var last_element = null;
var last_div = null;

var planets = ["1111", "2222","3333", "4444","5555","6666","7777","8888","9999" ];

function select_edge() {
    if ( last_element != null ) {
        last_element.style.stroke = "";
    }
    if ( last_div != null ) {
        last_div.style.display = "none";
    }
    this.style.stroke="red";
    last_element = this;
    
    var div_name = "details" + this.id;
    var el = document.getElementById(div_name);
    if ( el != null ) {
        el.style.display = "";
        last_div = el;
    } else {
        console.log( "Can't find div " + div_name );
    }
    
}

for (p1 of planets) {
    for (p2 of planets) {
        var name = "trade" + p1 + "x" + p2;
        var el = document.getElementById(name);
        if ( el != null ) {
            el.addEventListener('click', select_edge, false );
        }
    }
}
