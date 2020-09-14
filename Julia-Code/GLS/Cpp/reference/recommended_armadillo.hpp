struct recommended_armadillo
{
template<typename Type_X2, typename Type_S2, typename Type_y2>
decltype(auto) operator()(Type_X2 && X2, Type_S2 && S2, Type_y2 && y2)
{
    auto z2 = (arma::solve((X2).t()*arma::solve(S2, X2, arma::solve_opts::fast), (X2).t(), arma::solve_opts::fast)*arma::solve(S2, y2, arma::solve_opts::fast)).eval();

    typedef std::remove_reference_t<decltype(z2)> return_t;
    return return_t(z2);                         
}
};