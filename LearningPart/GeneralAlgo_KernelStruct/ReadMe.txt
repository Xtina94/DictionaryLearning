In this folder I modified the initial algorithm in two main ways:
1. I added to the graph learning part also the dictionary learning part, optimizing alternatevly between the two of them;
2. In the dictionary learniing part I didn't use the constraint over the kernels in the optimization function, but I used the Mathematical structure for the coefficients vector in the objective function;
3. I started the optimization supposing a random dictionary and opitimizing the alpha coefficients first;

Let's see if this gives a better result than starting the optimization cycle from the graph learning part;