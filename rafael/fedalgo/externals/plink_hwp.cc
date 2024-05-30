#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static const double kSmallEpsilon = 0.00000000000005684341886080801486968994140625;
static const double kExactTestBias = 0.00000000000000000000000010339757656912845935892608650874535669572651386260986328125;
static const double DBL_MAX = 1.7976931348623157e308;

double HweP(int32_t obs_hets, int32_t obs_hom1, int32_t obs_hom2, uint32_t midp) {
  // This function implements an exact SNP test of Hardy-Weinberg
  // Equilibrium as described in Wigginton, JE, Cutler, DJ, and
  // Abecasis, GR (2005) A Note on Exact Tests of Hardy-Weinberg
  // Equilibrium. American Journal of Human Genetics. 76: 887 - 893.
  //
  // The original version was written by Jan Wigginton.
  //
  // This version was written by Christopher Chang.  It contains the following
  // improvements over the original SNPHWE():
  // - Proper handling of >64k genotypes.  Previously, there was a potential
  //   integer overflow.
  // - Detection and efficient handling of floating point overflow and
  //   underflow.  E.g. instead of summing a tail all the way down, the loop
  //   stops once the latest increment underflows the partial sum's 53-bit
  //   precision; this results in a large speedup when max heterozygote count
  //   >1k.
  // - No malloc() call.  It's only necessary to keep track of a few partial
  //   sums.
  // - Support for the mid-p variant of this test.  See Graffelman J, Moreno V
  //   (2013) The mid p-value in exact tests for Hardy-Weinberg equilibrium.
  //
  // Note that the HweThresh() function below is a lot more efficient for
  // testing against a p-value inclusion threshold.  HweP() should only be
  // used if you need the actual p-value.
  intptr_t obs_homc;
  intptr_t obs_homr;
  if (obs_hom1 < obs_hom2) {
    obs_homc = obs_hom2;
    obs_homr = obs_hom1;
  } else {
    obs_homc = obs_hom1;
    obs_homr = obs_hom2;
  }
  const int64_t rare_copies = 2LL * obs_homr + obs_hets;
  const int64_t genotypes2 = (obs_hets + obs_homc + obs_homr) * 2LL;
  if (!genotypes2) {
    if (midp) {
      return 0.5;
    }
    return 1;
  }
  int32_t tie_ct = 1;
  double curr_hets_t2 = obs_hets;
  double curr_homr_t2 = obs_homr;
  double curr_homc_t2 = obs_homc;
  double tailp = (1 - kSmallEpsilon) * kExactTestBias;
  double centerp = 0;
  double lastp2 = tailp;
  double lastp1 = tailp;

  if (obs_hets * genotypes2 > rare_copies * (genotypes2 - rare_copies)) {
    // tail 1 = upper
    while (curr_hets_t2 > 1.5) {
      // het_probs[curr_hets] = 1
      // het_probs[curr_hets - 2] = het_probs[curr_hets] * curr_hets * (curr_hets - 1.0)
      curr_homr_t2 += 1;
      curr_homc_t2 += 1;
      lastp2 *= (curr_hets_t2 * (curr_hets_t2 - 1)) / (4 * curr_homr_t2 * curr_homc_t2);
      curr_hets_t2 -= 2;
      if (lastp2 < kExactTestBias) {
        tie_ct += (lastp2 > (1 - 2 * kSmallEpsilon) * kExactTestBias);
        tailp += lastp2;
        break;
      }
      centerp += lastp2;
      // doesn't seem to make a difference, but seems best to minimize use of
      // INFINITY
      if (centerp > DBL_MAX) {
        return 0;
      }
    }
    if ((centerp == 0) && (!midp)) {
      return 1;
    }
    while (curr_hets_t2 > 1.5) {
      curr_homr_t2 += 1;
      curr_homc_t2 += 1;
      lastp2 *= (curr_hets_t2 * (curr_hets_t2 - 1)) / (4 * curr_homr_t2 * curr_homc_t2);
      curr_hets_t2 -= 2;
      const double preaddp = tailp;
      tailp += lastp2;
      if (tailp <= preaddp) {
        break;
      }
    }
    double curr_hets_t1 = obs_hets + 2;
    double curr_homr_t1 = obs_homr;
    double curr_homc_t1 = obs_homc;
    while (curr_homr_t1 > 0.5) {
      // het_probs[curr_hets + 2] = het_probs[curr_hets] * 4 * curr_homr * curr_homc / ((curr_hets + 2) * (curr_hets + 1))
      lastp1 *= (4 * curr_homr_t1 * curr_homc_t1) / (curr_hets_t1 * (curr_hets_t1 - 1));
      const double preaddp = tailp;
      tailp += lastp1;
      if (tailp <= preaddp) {
        break;
      }
      curr_hets_t1 += 2;
      curr_homr_t1 -= 1;
      curr_homc_t1 -= 1;
    }
  } else {
    // tail 1 = lower
    while (curr_homr_t2 > 0.5) {
      curr_hets_t2 += 2;
      lastp2 *= (4 * curr_homr_t2 * curr_homc_t2) / (curr_hets_t2 * (curr_hets_t2 - 1));
      curr_homr_t2 -= 1;
      curr_homc_t2 -= 1;
      if (lastp2 < kExactTestBias) {
        tie_ct += (lastp2 > (1 - 2 * kSmallEpsilon) * kExactTestBias);
        tailp += lastp2;
        break;
      }
      centerp += lastp2;
      if (centerp > DBL_MAX) {
        return 0;
      }
    }
    if ((centerp == 0) && (!midp)) {
      return 1;
    }
    while (curr_homr_t2 > 0.5) {
      curr_hets_t2 += 2;
      lastp2 *= (4 * curr_homr_t2 * curr_homc_t2) / (curr_hets_t2 * (curr_hets_t2 - 1));
      curr_homr_t2 -= 1;
      curr_homc_t2 -= 1;
      const double preaddp = tailp;
      tailp += lastp2;
      if (tailp <= preaddp) {
        break;
      }
    }
    double curr_hets_t1 = obs_hets;
    double curr_homr_t1 = obs_homr;
    double curr_homc_t1 = obs_homc;
    while (curr_hets_t1 > 1.5) {
      curr_homr_t1 += 1;
      curr_homc_t1 += 1;
      lastp1 *= (curr_hets_t1 * (curr_hets_t1 - 1)) / (4 * curr_homr_t1 * curr_homc_t1);
      const double preaddp = tailp;
      tailp += lastp1;
      if (tailp <= preaddp) {
        break;
      }
      curr_hets_t1 -= 2;
    }
  }
  if (!midp) {
    return tailp / (tailp + centerp);
  }
  double pvalue = (tailp - ((1 - kSmallEpsilon) * kExactTestBias * 0.5) * tie_ct) / (tailp + centerp);
  printf("%f", pvalue); 
  return pvalue;
}

void HweP_vec (int32_t *obs_hets, int32_t *obs_hom1, int32_t *obs_hom2, double *pvalue, int ll, uint32_t midp ){
  for(int i = 0; i < ll; i++){
    pvalue[i] = HweP(obs_hets[i], obs_hom1[i], obs_hom2[i], midp);
  } 
}

int main() {
   // printf() displays the string inside quotation
  double pvalue = HweP(2882, 8710, 209, 0);
  printf("%f", pvalue); 
  return 0;
}

extern "C" {
    double HweP_py(int32_t obs_hets, int32_t obs_hom1, int32_t obs_hom2, uint32_t midp){ 
      double pvalue = HweP(obs_hets, obs_hom1, obs_hom2, midp); 
      return pvalue;
    }
    void HweP_vec_py(int32_t *obs_hets, int32_t *obs_hom1, int32_t *obs_hom2, double *pvalue, int ll, uint32_t midp){
      HweP_vec(obs_hets, obs_hom1, obs_hom2, pvalue, ll, midp);
    }
}
