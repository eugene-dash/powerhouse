# powerhouse
<br>A wrapper for cupy and numpy that allows programs to utilize Cupy in it's presence and still work in it's absence.<br><br>
Currently, powerhouse soley exists as a very simple band-aid sollution to allow me to develop personal gpu utilizing libraries on machines with or without cuda<br><br>
My goal is to at some point, condense both Cupy and Numpy APIs into a single API.<br>
This single API, in theory, should simply replace the Cupy side of it with Numpy when Cupy is not present.<br>
