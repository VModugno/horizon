#include "sqp.h"

int main()
{
    int N = 3;

    auto x = casadi::SX::sym("x", N+1);
    auto u = casadi::SX::sym("u", N);
    auto ps = casadi::SX::sym("ps", 2);//["p", "dt"]

    std::vector<casadi::Matrix<casadi::SXElem>> f_vec;
    for(unsigned int i = 0; i < N; ++i)
        f_vec.push_back(x(i+1) - (x(i) + ps(1)*u(i)));
    auto f = casadi::SX::vertcat(f_vec);
    std::cout<<"f: "<<f<<std::endl;

    std::vector<casadi::Matrix<casadi::SXElem>> cf_vec;
    cf_vec.push_back(x(2) - ps(0));
    auto cf = casadi::SX::vertcat(cf_vec);
    std::cout<<"cf: "<<cf<<std::endl;

    std::vector<casadi::Matrix<casadi::SXElem>> l_vec;
    l_vec.push_back(u(0)*1e-3);
    l_vec.push_back(u(1)*1e-3);
    auto ll = casadi::SX::vertcat(l_vec);
    std::cout<<"ll: "<<ll<<std::endl;

    //auto l = casadi::Function("l", {casadi::SX::vertcat({x,u}), lp}, {ll}, {"x", "lp"}, {"l"});
    auto l = casadi::Function("l", {casadi::SX::vertcat({x,u}), ps}, {ll}, {"x", "ps"}, {"l"});

//    auto g = casadi::Function("g", {casadi::SX::vertcat({x,u}), dt, p},
//                              {casadi::SX::vertcat({f, cf})},
//                              {"x", "dt", "p"}, {"g"});
    auto g = casadi::Function("g", {casadi::SX::vertcat({x,u}), ps},
                              {casadi::SX::vertcat({f, cf})},
                              {"x", "ps"}, {"g"});


    casadi::Dict opts;
    opts["max_iter"] = 10;
    horizon::SQPGaussNewton<casadi::SX> sqp("sqp", "qpoases", l, g, opts);

    std::cout<<"x size1: "<<x.size1()<<std::endl;

    casadi::DM lb(x.size1() + u.size1(), 1);
    casadi::DM ub(x.size1() + u.size1(), 1);
    for(unsigned int i = 0; i < lb.size1(); ++i)
    {
        lb(i) = -INFINITY;
        ub(i) = INFINITY;
    }

    lb(N) = 1.;
    ub(N) = 1.;

    lb(N+1) = 11.;
    ub(N+1) = 11.;

    std::cout<<"lb: "<<lb<<std::endl;
    std::cout<<"ub: "<<ub<<std::endl;

    casadi::DM x0(2*N+1, 1);
    for(unsigned int i = 0; i < x0.size1(); ++i)
        x0(i) = 0.;

    casadi::DM lg(f.size1() + cf.size1(), 1);
    casadi::DM ug(f.size1() + cf.size1(), 1);
    for(unsigned int i = 0; i < lg.size1(); ++i)
    {
        lg(i) = 0.;
        ug(i) = 0.;
    }
    std::cout<<"lg: "<<lg<<std::endl;
    std::cout<<"ug: "<<ug<<std::endl;


    casadi::DM ps_val(2,1);
    ps_val(0) = -1.;
    ps_val(1) = 0.1;

    auto solution = sqp.solve(x0, ps_val, lb, ub, lg, ug);
    std::cout<<"solution: "<<solution["x"]<<"   f: "<<solution["f"]<<"  g: "<<solution["g"]<<std::endl;
}
