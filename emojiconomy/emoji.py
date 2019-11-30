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
    "🔥" : "FIRE",
    "🔪" : "KITCHEN KNIFE",
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
    "🏗️" : "CONSTRUCTION CRANE",
    "🚧" : "CONSTRUCTION",
    "👷🏿" : "CONSTRUCTION WORKER",
    "👩‍⚕️" : "BIO WORKER",
    "👩‍💼" : "OFFICE WORKER",
    "👩🏽‍🏭" : "WOMAN FACTORY WORKER",
    "🏦" : "BANK",
    "🏢" : "OFFICE BUILDING",
    "📈" : "CHART INCREASING",
    "👩‍🚒" : "FIREFIGHTER",
    "⚡" : "HIGH VOLTAGE",
    "🕰️" : "CLOCK",
    "⏳" : "HOURGLASS",
    "⏲️" : "TIMER CLOCK",
    "🍳" : "FRYING PAN",
    "👨‍🍳" : "MAN COOK",
    "👩‍🍳" : "WOMAN COOK",
    "👻" : "GHOST",
    "👾" : "ALIEN MONSTER",
    "👽" : "ALIEN",
    "🧑🏾‍🚀" : "ASTRONAUT",
    "🧛" : "VAMPIRE",
    "😱" : "SCREAM"
}

ores = {
    "🗻" : "MOUNT FUJI",
    "⛰" : "MOUNTAIN",
    "🏔️" : "SNOWCAPPED MOUNTAIN",
    "🌋" : "LAVA",
    "☿" : "MERCURY",
    "🌑" : "MOON",
    "⚷" : "CHIRON",
    "☢" : "RADIOACTIVE",
    "🌙" : "CRESCENT MOON",
    "☄" : "COMET",
    "🜜" : "IRON ORE",
}

ore_processed = {
    "💎" : "GEM",
    "♦️" : "DIAMOND",
    "💠" : "DIAMOND WITH DOT",
    "🔶" : "ORANGE DIAMOND",
    "🔮" : "CRYSTAL BALL",
    "🧱" : "BRICK"
}

metals = {
    "⚴" : "PALLADIUM",
    "⛎" : "OPHIUCHUS",
    "🜏" : "SULPHUR",
    "🜚" : "ALECHEMICAL GOLD",
    "🥇" : "GOLD",
    "🜟" : "REGULUS OF IRON",
    "㊎" : "METAL",
    "Fe" : "IRON",
    "Ti" : "TITANIUM",
    "Pb" : "LEAD",
    "Be" : "BERYLLIUM",
    "Au" : "ELEMENTAL GOLD",
    "Ag" : "ELEMENTAL SILVER",
    "⬛" : "BLACK SQUARE",
}

metal_parts = {
    "⚙️" : "GEAR",
    "💡" : "LIGHT BULB",
    "🔋" : "BATTERY",
    "🧲" : "MAGNET",
    "🔩" : "NUT AND BOLT"
}

machines = {
    "💻" : "COMPUTER",
    "💾" : "FLOPPY DISK",
    "💽" : "COMPUTER DISK",
    "🎹" : "MUSICAL KEYBOARD",
    "📢" : "LOUDSPEAKER",
    "📶" : "ANTENNA BARS",
    "📺" : "TELEVISON",
    "🎙️" : "STUDIO MICROPHONE",
}

housing = {
    "🏘️" : "HOUSES",
    "🏠" : "HOUSE",
    "🛏️" : "BED",
    "🛋️" : "SOFA",
    "🏨" : "HOTEL"
}

toys = {
    "🧸" : "TEDDY BEAR",
    "🎮" : "VIDEO GAME",
    "⚽" : "SOCCER BALL",
    "⚾" : "BASEBALL",
}

vehicles = {
    "🚗" : "AUTOMOBILE",
    "🚙" : "SPORT UTILITY VEHICLE",
    "🏎️" : "RACING CAR",
    "️🛥️" : "MOTOR BOAT",
}

full_dict = {}
full_dict.update( plants )
full_dict.update( plant_processed )
full_dict.update( plant_products )
full_dict.update( foods )
full_dict.update( processes )
full_dict.update( factories )
full_dict.update( ores )
full_dict.update( ore_processed )
full_dict.update( metals )
full_dict.update( metal_parts )
full_dict.update( machines )
full_dict.update( housing )
full_dict.update( toys )
full_dict.update( vehicles )

def emojify_graph( g ):
    remaining = {
        "plant" : set( plants.keys() ),
        "plant_proc" : set( plant_processed.keys() ),
        "plant_prod" : set( plant_products.keys() ),
        "food" : set( foods.keys() ),
        "process" : set( processes.keys() ),
        "factory" : set( factories.keys() ),
        "ore" : set( ores.keys() ),
        "ore_proc" : set( ore_processed.keys() ),
        "metal" : set( metals.keys() ),
        "metal_part" : set( metal_parts.keys() ),
        "machine" : set( machines.keys() ),
        "housing" : set( housing.keys() ),
        "toy" : set( toys.keys() ),
        "vehicle" : set( vehicles.keys() )
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
            


