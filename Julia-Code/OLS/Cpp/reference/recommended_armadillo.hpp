struct recommended_armadillo
{
template<typename Type_X1, typename Type_y1>
decltype(auto) operator()(Type_X1 && X1, Type_y1 && y1)
{
    auto b1 = (arma::solve((X1).t()*X1, (X1).t(), arma::solve_opts::fast)*y1).eval();

    typedef std::remove_reference_t<decltype(b1)> return_t;
    return return_t(b1);                         
}
};