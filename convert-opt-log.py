print( ",".join( ["gradient_time", "gradient_iter", "gradient_success",
                  "augment_time", "augment_iter", "augment_big", "augment_hist" ] ) )

with open( "optimization.log", "r" ) as f:
    try:
        while True:
            l1 = next( f )
            l2 = next( f )
            tok1 = l1.split( " " )
            tok2 = l2.split( " " )
            assert tok1[0] == "gradient"
            assert tok2[0] == "augment"
            print( ",".join( [tok1[1], tok1[2], tok1[3],
                              tok2[1], tok2[2], tok2[3], tok2[4]] ) )
    except StopIteration:
        pass
        
        
