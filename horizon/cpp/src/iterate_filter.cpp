#include "iterate_filter.h"
#include <math.h>

IterateFilter::Pair::Pair():
    f(std::numeric_limits<double>::max()),
    h(std::numeric_limits<double>::max())
{}

bool IterateFilter::Pair::dominates(const IterateFilter::Pair &other, double beta, double gamma) const
{
    bool ret = false;

    if(isinf(other.f) || isinf(other.h))
    {
        ret = true;
    }
    else
    {
        ret = f < other.f + gamma*other.h && beta*h < other.h;
    }

    // {
    //     printf("(%e, %e) %s dominate (%e, %e) \n", 
    //             f, h, ret ? "" : "does not", other.f, other.h);
    // }

    return ret;
}

bool IterateFilter::is_acceptable(const IterateFilter::Pair &test_pair) const
{

    auto dominates_test_pair = [&test_pair, this](const Pair& filt_pair)
    {
        return filt_pair.dominates(test_pair, beta, gamma);
    };


    int ndom = std::count_if(_entries.begin(),
                             _entries.end(),
                             dominates_test_pair);

    // std::cout << "ndom = " << ndom << ", memory is " << memory << "\n";

    return ndom <= memory;
}

bool IterateFilter::add(const IterateFilter::Pair &new_pair)
{
    // if new pair is not accepted, just return false
    if(!is_acceptable(new_pair))
    {
        return false;
    }

    // count how many pairs in the filter are dominated by the newly
    // accepted pair
    auto dominated_by_new_pair = [&new_pair](const Pair& filt_pair)
    {
        return new_pair.dominates(filt_pair);
    };

    int ndom = std::count_if(_entries.begin(),
                             _entries.end(),
                             dominated_by_new_pair);
    
    // we need to remove them, and leave just $memory items (non-monotone filter)
    // see Nonmonotone Filter Method for Nonlinear Optimization (Roger Fletcher, Sven Leyffer, and Chungen Shen)
    int nremoved = 0;
    int toremove = std::max(0, ndom - memory);

    // std::cout << "will remove " << toremove << " pairs (memory is " << memory << "\n";
    
    auto dominated_by_new_pair_memory = [&new_pair, &nremoved, toremove](const Pair& filt_pair)
    {
        bool dom = new_pair.dominates(filt_pair);
        if(dom && nremoved < toremove)
        {
            ++nremoved;
            
            return true;
        }
        return false;
    };

    _entries.remove_if(dominated_by_new_pair_memory);

    // push back new pair
    // note: we never add a pair with too low constraint value, 
    // since this would hinder the cost from decreasing while
    // staying inside constr violation tolerances
    auto new_pair_copy = new_pair;
    
    new_pair_copy.h = std::max(constr_tol, new_pair.h);

    _entries.push_back(new_pair_copy);

    return true;
}

void IterateFilter::clear()
{
    _entries.clear();
}

void IterateFilter::print()
{
    for(const auto& ent : _entries)
    {
        printf("(%4.3e, %4.3e) \n", ent.f, ent.h);
    }
}
