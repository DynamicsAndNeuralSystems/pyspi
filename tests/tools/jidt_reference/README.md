# JIDT reference values

`JidtReference.java` produced the constants hard-coded in
`tests/test_jidt_reference.py`. pyspi 2.x computed its information-theoretic
measures with JIDT; pyspi 3 reimplements them in NumPy, and the test keeps the
evidence that the port reproduces JIDT without needing Java at test time.

JIDT (Java Information Dynamics Toolkit; Lizier, 2014, *Front. Robot. AI* 1:11) is
GPL-3.0-or-later, https://github.com/jlizier/jidt. The jar is **not** committed;
fetch version 1.6.1 (the one the constants came from) with either of:

```sh
# the copy shipped inside the pyspi 2.0.2 sdist/wheel
pip download pyspi==2.0.2 --no-deps -d /tmp/pyspi202 && unzip -j /tmp/pyspi202/pyspi-2.0.2-*.whl 'pyspi/lib/jidt/infodynamics.jar' -d /tmp
# or from the upstream release: https://github.com/jlizier/jidt/releases/tag/v1.6.1
```

Then, from this directory (Java 11 was used):

```sh
javac -cp /tmp/infodynamics.jar -d /tmp/jr JidtReference.java
java  -cp /tmp/infodynamics.jar:/tmp/jr JidtReference ../../data/fixtures/jidt_ar_pair_T500.csv
```

Each output line is `name value`. Kernel and symbolic values are in bits, the rest in
nats; the test converts pyspi's nats with `/ ln 2` where needed. Paste the values into
the `JIDT` dictionary in the test. The input is the first 500 rows of a coupled
AR(1) pair (`x[t] = 0.5 x[t-1] + e`, `y[t] = 0.4 y[t-1] + 0.6 x[t-1] + e`,
numpy `default_rng(1)`), written with `%.17g` so Java and NumPy read identical doubles.
