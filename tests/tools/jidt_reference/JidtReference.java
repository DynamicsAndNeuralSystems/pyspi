// Generates the JIDT 1.6.1 reference values hard-coded in tests/test_jidt_reference.py.
//
//   javac -cp infodynamics.jar JidtReference.java
//   java  -cp infodynamics.jar:. JidtReference ../../data/fixtures/jidt_ar_pair_T500.csv
//
// Input: a CSV with one row per time step and two columns (x, y); the estimators
// compute information flow x -> y. Output: one "name value" line per quantity.
// Kernel and symbolic quantities are in JIDT's base 2 (bits); everything else in nats.
// JIDT is GPL-3.0-or-later (https://github.com/jlizier/jidt).
import java.io.*;
import java.util.*;
import infodynamics.measures.continuous.gaussian.*;
import infodynamics.measures.continuous.kernel.*;
import infodynamics.measures.continuous.kraskov.*;
import infodynamics.measures.continuous.symbolic.*;

public class JidtReference {
  static double[][] read(String f) throws Exception {
    List<double[]> rows = new ArrayList<>();
    try (BufferedReader r = new BufferedReader(new FileReader(f))) {
      String s;
      while ((s = r.readLine()) != null) {
        String[] p = s.split(",");
        double[] d = new double[p.length];
        for (int i = 0; i < p.length; i++) d[i] = Double.parseDouble(p[i]);
        rows.add(d);
      }
    }
    return rows.toArray(new double[0][]);
  }

  static double[] col(double[][] a, int c) {
    double[] o = new double[a.length];
    for (int i = 0; i < a.length; i++) o[i] = a[i][c];
    return o;
  }

  static void out(String name, double v) { System.out.println(name + " " + v); }

  public static void main(String[] args) throws Exception {
    double[][] d = read(args[0]);
    double[] x = col(d, 0), y = col(d, 1);
    int T = x.length;
    String K = "4";

    // Gaussian MI and time-lagged MI (lag 1)
    MutualInfoCalculatorMultiVariateGaussian g = new MutualInfoCalculatorMultiVariateGaussian();
    g.initialise(1, 1); g.setObservations(x, y);
    out("mi_gaussian", g.computeAverageLocalOfObservations());
    g = new MutualInfoCalculatorMultiVariateGaussian();
    g.initialise(1, 1);
    g.setObservations(Arrays.copyOfRange(x, 0, T - 1), Arrays.copyOfRange(y, 1, T));
    out("tlmi_gaussian", g.computeAverageLocalOfObservations());

    // Kernel MI (bits)
    MutualInfoCalculatorMultiVariateKernel km = new MutualInfoCalculatorMultiVariateKernel();
    km.setProperty("KERNEL_WIDTH", "0.5"); km.initialise(1, 1); km.setObservations(x, y);
    out("mi_kernel_bits", km.computeAverageLocalOfObservations());

    // KSG (algorithm 1) MI at Theiler windows 0, 5, 20. No dither: JIDT's default
    // NOISE_LEVEL_TO_ADD (1e-8) would make the reference stochastic, so it is set to 0.
    // NORMALISE stays at its default (true).
    for (int w : new int[]{0, 5, 20}) {
      MutualInfoCalculatorMultiVariateKraskov1 kr = new MutualInfoCalculatorMultiVariateKraskov1();
      kr.setProperty("k", K); kr.setProperty("NOISE_LEVEL_TO_ADD", "0");
      kr.setProperty("DYN_CORR_EXCL", "" + w);
      kr.initialise(1, 1); kr.setObservations(x, y);
      out("mi_ksg_w" + w, kr.computeAverageLocalOfObservations());
    }

    // Gaussian TE
    for (int kh : new int[]{1, 3}) {
      TransferEntropyCalculatorGaussian te = new TransferEntropyCalculatorGaussian();
      te.setProperty("k_HISTORY", "" + kh); te.initialise(); te.setObservations(x, y);
      out("te_gaussian_k" + kh, te.computeAverageLocalOfObservations());
    }
    { TransferEntropyCalculatorGaussian te = new TransferEntropyCalculatorGaussian();
      te.setProperty("k_HISTORY", "2"); te.setProperty("k_TAU", "2");
      te.setProperty("l_HISTORY", "3"); te.setProperty("l_TAU", "2");
      te.initialise(); te.setObservations(x, y);
      out("te_gaussian_k2t2_l3t2", te.computeAverageLocalOfObservations()); }

    // KSG TE: fixed embeddings
    for (int kh : new int[]{1, 3}) {
      TransferEntropyCalculatorKraskov te = new TransferEntropyCalculatorKraskov();
      te.setProperty("k", K); te.setProperty("NOISE_LEVEL_TO_ADD", "0");
      te.setProperty("k_HISTORY", "" + kh); te.initialise(); te.setObservations(x, y);
      out("te_ksg_k" + kh, te.computeAverageLocalOfObservations());
    }
    { TransferEntropyCalculatorKraskov te = new TransferEntropyCalculatorKraskov();
      te.setProperty("k", K); te.setProperty("NOISE_LEVEL_TO_ADD", "0");
      te.setProperty("k_HISTORY", "2"); te.setProperty("k_TAU", "2");
      te.setProperty("l_HISTORY", "3"); te.setProperty("l_TAU", "2");
      te.initialise(); te.setObservations(x, y);
      out("te_ksg_k2t2_l3t2", te.computeAverageLocalOfObservations()); }
    // KSG TE with dynamic correlation exclusion (Theiler window 10)
    { TransferEntropyCalculatorKraskov te = new TransferEntropyCalculatorKraskov();
      te.setProperty("k", K); te.setProperty("NOISE_LEVEL_TO_ADD", "0");
      te.setProperty("k_HISTORY", "1"); te.setProperty("DYN_CORR_EXCL", "10");
      te.initialise(); te.setObservations(x, y);
      out("te_ksg_k1_dce10", te.computeAverageLocalOfObservations()); }
    // KSG TE with MAX_CORR_AIS auto-embedding and a Theiler window of 10 (DCE).
    // JIDT's AUTO_EMBED_* search is left to pick the embedding; it is printed too.
    { TransferEntropyCalculatorKraskov te = new TransferEntropyCalculatorKraskov();
      te.setProperty("k", K); te.setProperty("NOISE_LEVEL_TO_ADD", "0");
      te.setProperty("DYN_CORR_EXCL", "10");
      te.setProperty("AUTO_EMBED_METHOD", "MAX_CORR_AIS");
      te.setProperty("AUTO_EMBED_K_SEARCH_MAX", "10");
      te.setProperty("AUTO_EMBED_TAU_SEARCH_MAX", "4");
      te.initialise(); te.setObservations(x, y);
      out("te_ksg_dce10_auto", te.computeAverageLocalOfObservations());
      System.out.println("te_ksg_dce10_auto_embedding k=" + te.getProperty("k_HISTORY")
          + " ktau=" + te.getProperty("k_TAU") + " l=" + te.getProperty("l_HISTORY")
          + " ltau=" + te.getProperty("l_TAU")); }
    // Gaussian TE with MAX_CORR_AIS auto-embedding
    { TransferEntropyCalculatorGaussian te = new TransferEntropyCalculatorGaussian();
      te.setProperty("AUTO_EMBED_METHOD", "MAX_CORR_AIS");
      te.setProperty("AUTO_EMBED_K_SEARCH_MAX", "10");
      te.setProperty("AUTO_EMBED_TAU_SEARCH_MAX", "4");
      te.initialise(); te.setObservations(x, y);
      out("te_gaussian_auto", te.computeAverageLocalOfObservations());
      System.out.println("te_gaussian_auto_embedding k=" + te.getProperty("k_HISTORY")
          + " ktau=" + te.getProperty("k_TAU") + " l=" + te.getProperty("l_HISTORY")
          + " ltau=" + te.getProperty("l_TAU")); }

    // Kernel TE (bits). Not reproduced exactly by pyspi; see the test.
    for (int kh : new int[]{1, 2}) {
      TransferEntropyCalculatorKernel te = new TransferEntropyCalculatorKernel();
      te.setProperty("KERNEL_WIDTH", "0.5"); te.setProperty("k_HISTORY", "" + kh);
      te.initialise(); te.setObservations(x, y);
      out("te_kernel_bits_k" + kh, te.computeAverageLocalOfObservations());
    }

    // Symbolic TE (ordinal patterns of length k; bits)
    for (int kh : new int[]{2, 3}) {
      TransferEntropyCalculatorSymbolic te = new TransferEntropyCalculatorSymbolic();
      te.setProperty("k_HISTORY", "" + kh); te.initialise(); te.setObservations(x, y);
      out("te_symbolic_k" + kh, te.computeAverageLocalOfObservations());
    }
  }
}
