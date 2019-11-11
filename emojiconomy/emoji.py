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
    "🌿" : "HERB",
    "🌾" : "SHEAF OF RICE",
}

plant_processed = {
    "🍞" : "BREAD",
    "🛢️" : "OIL DRUM",
    "🧪" : "TEST TUBE",
    "🦠" : "MICROBE",
    "🧃" : "BEVERAGE BOX",
    "🥤" : "CUP",
    "🍚" : "COOKED_RICE"
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

processes = {
    "🎚️" : "LEVEL SLIDER",
    "🎛️" : "CONTROL KNOBS",
    "🪓" : "AXE",
    "🧰" : "TOOLBOX",
    "⚗️" : "ALEMBIC",
    "🧙" : "MAGE",
    "🎰" : "SLOT MACHINE",
    "🧺" : "BASKET",
    "️⚛️" : "ATOM",
}

factories = {
    "👨‍🏭" : "MAN WORKER",
    "👩‍🏭" : "WOMAN WORKER",
    "🏭" : "FACTORY",
    "🏬" : "DEPARTMENT STORE",
    "🧩" : "PUZZLE PIECE",
    "🕘🕔" : "NINE TO FIVE",
    "♊" : "GEMINI",
    "🤖" : "ROBOT",
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

full_dict = {}
full_dict.update( plants )
full_dict.update( plant_processed )
full_dict.update( plant_products )
full_dict.update( foods )
full_dict.update( processes )
full_dict.update( factories )

def emojify_graph( g ):
    remaining = {
        "plant" : set( plants.keys() ),
        "plant_proc" : set( plant_processed.keys() ),
        "plant_prod" : set( plant_products.keys() ),
        "food" : set( foods.keys() ),
        "process" : set( processes.keys() ),
        "factory" : set( factories.keys() ),
    }

    for n, category in list( g.nodes.data( 'tag' ) ):
        if category in remaining and len( remaining[category] ) > 0:
            e = random.choice( list( remaining[category] ) )
            g.nodes[n]['tag'] = e
            g.nodes[n]['text'] = full_dict[e]
            g.nodes[n]['category'] = category
            remaining[category].remove( e )
        else:
            g.nodes[n]['text'] = "unassigned"
            g.nodes[n]['category'] = category
            


