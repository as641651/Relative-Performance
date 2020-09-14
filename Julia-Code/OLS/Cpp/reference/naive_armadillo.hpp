struct naive_armadillo
{
template<typename Type_X1, typename Type_y1>
decltype(auto) operator()(Type_X1 && X1, Type_y1 && y1)
{
    auto b1 = (((X1).t()*X1).i()*(X1).t()*y1).eval();

    typedef std::remove_reference_t<decltype(b1)> return_t;
    return return_t(b1);                         
}
};