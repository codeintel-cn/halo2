fn main() {
    // use blstrs::{Fp, Fp2, G1Affine, G2Affine, Scalar};
    use ec_gpu_gen::SourceBuilder;
    use halo2curves::bn256::{Fq, Fq2, G1Affine, G2Affine, Fr};
    use halo2curves::pasta::{EpAffine, Ep, EqAffine, Eq};

    let source_builder = SourceBuilder::new()
        .add_fft::<Fr>()
        .add_multiexp::<G1Affine, Fq>()
        // .add_multiexp::<EpAffine, Ep>()
        // .add_multiexp::<EqAffine, Eq>()
        .add_multiexp::<G2Affine, Fq2>();
    ec_gpu_gen::generate(&source_builder);
}
