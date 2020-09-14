struct naive_armadillo
{
template<typename Type_X2, typename Type_S2, typename Type_y2>
decltype(auto) operator()(Type_X2 && X2, Type_S2 && S2, Type_y2 && y2)
{
    auto z2 = (((X2).t()*arma::inv_sympd(S2)*X2).i()*(X2).t()*arma::inv_sympd(S2)*y2).eval();

    typedef std::remove_reference_t<decltype(z2)> return_t;
    return return_t(z2);                         
}
};