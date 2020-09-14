#include <generator/generator.hpp>

template<typename Gen>
decltype(auto) operand_generator(Gen && gen)
{
    auto X1 = gen.generate({1000,500}, generator::property::random{}, generator::shape::not_square{});
    auto y1 = gen.generate({1000,1}, generator::property::random{}, generator::shape::col_vector{}, generator::shape::not_square{});
    return std::make_tuple(X1, y1);
}