type range = Step of float * float * float | Count of float * float * int

val mgrid : range list -> Types.value
val ogrid : range list -> Types.value list
val r_ : Types.value list -> Types.value
val c_ : Types.value list -> Types.value
