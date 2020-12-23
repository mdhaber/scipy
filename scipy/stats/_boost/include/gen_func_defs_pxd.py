'''Generate func_defs.pxd'''

import pathlib


def gen_func_defs_pxd(outfile, max_num_inputs=4):
    '''
    Cython does not support template parameter packs, so to keep it
    from freaking out, we'll manually produce all the different template
    expansions we need to call in the cython wrappers.
    '''

    contents = ''
    hdr = str((pathlib.Path(__file__).parent / "func_defs.hpp").as_posix())
    contents += f'cdef extern from "{hdr}" namespace "" nogil:\n'

    x_funcs = ('pdf', 'cdf', 'icdf', 'quantile', 'iquantile')  # functions that take ctor params and parameter "x"
    no_x_funcs = ('mean', 'variance', 'skewness', 'kurtosis_excess')  # functions that take only ctor params
    for ii in range(1, max_num_inputs+1):
        template_args = ', '.join(f'T{jj} arg{jj}' for jj in range(1, ii+1))
        template_types = ', '.join(f'T{jj}' for jj in range(1, ii+1))

        # for all the different "overloads", we need to produce a distinct Cython reference;
        # assumes that all number template types are the same, i.e. RealType == T1 == T2 == etc
        for func in x_funcs:
            fname = f'boost_{func}'
            contents += f'    RealType {fname}{ii} "{fname}" [Dist, RealType, {template_types}](RealType x, {template_args})\n'
        for func in no_x_funcs:
            fname = f'boost_{func}'
            contents += f'    RealType {fname}{ii} "{fname}" [Dist, RealType, {template_types}]({template_args})\n'

    # patch for boost::math::beta_distibution x=1, beta<1
    contents += f'    RealType boost_pdf_beta2 "boost_pdf_beta" [Dist, RealType, T1, T2](RealType x, T1 a, T2 b)'

    with open(outfile, 'w') as fp:
        fp.write(contents)


if __name__ == '__main__':
    pass
