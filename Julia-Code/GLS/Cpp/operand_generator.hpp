#include <generator/generator.hpp>

template<typename Gen>
decltype(auto) operand_generator(Gen && gen)
{
    auto X2 = gen.generate({250,500}, generator::property::random{}, generator::shape::not_square{});
    auto S2 = gen.generate({250,250}, generator::shape::self_adjoint{}, generator::property::spd{});
    auto y2 = gen.generate({250,1}, generator::property::random{}, generator::shape::col_vector{}, generator::shape::not_square{});
    return std::make_tuple(X2, S2, y2);
}