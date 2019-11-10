import random

plants = {
    "🌸" : "CHERRY BLOSSOM",
    "🍇" : "GRAPES",
    "🍈" : "MELON",
    "🍉" : "WATERMELON",
    "🍊" : "TANGERINE",
    "🍋" : "LEMON",
    "🍌" : "BANANA",
    "🌹" : "ROSE",
    "🌷" : "TULIP",
    "🌱" : "SEEDLING",
    "🌴" : "PALM",
    "🌲" : "EVERGREEN",
    "🌿" : "HERB"
}

plant_processed = {
    "🍞" : "BREAD",
    "🛢️" : "OIL DRUM",
    "🧪" : "TEST TUBE",
    "🦠" : "MICROBE",
    "🧃" : "BEVERAGE BOX",
    "🥤" : "CUP"
}

plant_products = {
    "🧶" : "YARN",
    "🧴" : "LOTION BOTTLE",
    "✨" : "SPARKLE",
    "🧻" : "PAPER",
}

foods = {
    "🥧" : "PIE",
    "🥟" : "DUMPLING",
    "🥫" : "CANNED FOOD",
    "🌮" : "TACO",
    "🍷" : "WINE",
    "🍔" : "HAMBURGER"
    
}

ores = {
}

ores_processed = {
}

machines = {
}

housing = {
}

toys = {
}

def emojify_graph( g ):
    remaining = {
        "plant" : set( plants.keys() ),
        "plant_proc" : set( plant_processed.keys() ),
        "plant_prod" : set( plant_products.keys() ),
        "food" : set( foods.keys() )
    }

    for n, category in list( g.nodes.data( 'tag' ) ):
        if category in remaining and len( remaining[category] ) > 0:
            e = random.choice( list( remaining[category] ) )
            g.nodes[n]['tag'] = e
            remaining[category].remove( e )


