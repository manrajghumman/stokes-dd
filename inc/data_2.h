/* ---------------------------------------------------------------------
 * Functions representing RHS, physical parameters, boundary conditions and
 * the true solution.
 * ---------------------------------------------------------------------
 *
 * Author: Manraj Singh Ghumman, University of Pittsburgh, 2023 - 2024
 */

#ifndef STOKES_MFEDD_DATA_H
#define STOKES_MFEDD_DATA_H

#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>

namespace dd_stokes
{
  using namespace dealii;

  template <int dim>
  class BoundaryValues : public Function<dim>
  {
  public:
    BoundaryValues()
      : Function<dim>(dim + 1)
    {}

    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override;

    virtual void vector_value(const Point<dim> &p,
                              Vector<double> &  value) const override;
  };


  template <int dim>
  double BoundaryValues<dim>::value(const Point<dim> & p,
                                    const unsigned int component) const
  {
    Assert(component < this->n_components,
           ExcIndexRange(component, 0, this->n_components));

    if (component == 0)//i.e. velocity in x direction u_x
      {
        return -3*p[0]+std::cos(M_PI*p[1]);
      }
    if (component == 1)//i.e. velocity in y direction u_y
      {
        return p[1]+std::sin(2*M_PI*p[0]);
      }

    return 0;
  }


  template <int dim>
  void BoundaryValues<dim>::vector_value(const Point<dim> &p,
                                         Vector<double> &  values) const
  {
    for (unsigned int c = 0; c < this->n_components-1; ++c)
      values(c) = BoundaryValues<dim>::value(p, c);
  }


  template <int dim>
    class StressTensor_Exact : public TensorFunction<2, dim>
  {
  public:
    StressTensor_Exact()
      : TensorFunction<2, dim>()
    {}
    //for F
    //function to explain the relation y = f(x)
    virtual Tensor<2, dim> value(const Point<dim> &p) const override;
  };

  template <int dim>
    Tensor<2,dim> StressTensor_Exact<dim>::value(const Point<dim> &p) const
  { Tensor<2, dim> return_value;
    {
    return_value [0][0] = -std::sin(M_PI*p[0])*std::cos(2*M_PI*p[1])-6;
    return_value [0][1] = M_PI*(2*std::cos(2*M_PI*p[0])-std::sin(M_PI*p[1]));
    return_value [1][0] = M_PI*(2*std::cos(2*M_PI*p[0])-std::sin(M_PI*p[1]));
    return_value [1][1] = -std::sin(M_PI*p[0])*std::cos(2*M_PI*p[1])+2;
    }

    return return_value;
  }

  template <int dim>
    class NormalStressTensor_Exact : public Function<dim>
  {
  public:
    NormalStressTensor_Exact(int face)
      : Function<dim>(dim+1), face(face)
    {}
    //for F
    //function to explain the relation y = f(x)
    virtual double value(const Point<dim> &p,
                        const unsigned int component) const override;

    virtual void vector_value(const Point<dim> &p,
                              Vector<double> &  value) const override;

    private:
    int face; // Determines which function behavior to use
  };

  template <int dim>
    double NormalStressTensor_Exact<dim>::value(const Point<dim> &p,
                                                const unsigned int component) const
  { 
    Assert(component < this->n_components,
           ExcIndexRange(component, 0, this->n_components));

    Tensor<2, dim> tensor_value;
    tensor_value [0][0] = -std::sin(M_PI*p[0])*std::cos(2*M_PI*p[1])-6;
    tensor_value [0][1] = M_PI*(2*std::cos(2*M_PI*p[0])-std::sin(M_PI*p[1]));
    tensor_value [1][0] = M_PI*(2*std::cos(2*M_PI*p[0])-std::sin(M_PI*p[1]));
    tensor_value [1][1] = -std::sin(M_PI*p[0])*std::cos(2*M_PI*p[1])+2;
    
    Tensor<1, dim> normal_direction;
    
    if (face == 0)
    {
      normal_direction[0] = 0;
      normal_direction[1] = -1;
    }
    else if (face == 1)
    {
      normal_direction[0] = 1;
      normal_direction[1] = 0;
    }
    else if (face == 2)
    {
      normal_direction[0] = 0;
      normal_direction[1] = 1;
    }
    else 
    {
      normal_direction[0] = -1;
      normal_direction[1] = 0;
    }
      
    if (component == 0)//i.e. normal_stress in x direction Tn_x
      return tensor_value[0][0] * fabs(normal_direction[0]) + tensor_value[0][1] * fabs(normal_direction[1]);

    if (component == 1)//i.e. normal_stress in y direction Tn_y
      return tensor_value[1][0] * fabs(normal_direction[0]) + tensor_value[1][1] * fabs(normal_direction[1]);
    
    if (component == 2)//dummy useless
      return 0;

    return 0;
  }

  template <int dim>
  void NormalStressTensor_Exact<dim>::vector_value(const Point<dim> &p,
                                                   Vector<double> &  values) const
  {
    for (unsigned int c = 0; c < this->n_components; ++c)
      values(c) = NormalStressTensor_Exact<dim>::value(p, c);
  }


  template <int dim>
  class RightHandSide : public TensorFunction<1, dim>
  {
  public:
    RightHandSide()
      : TensorFunction<1, dim>()
    {}
    //for F
    //function to explain the relation y = f(x)
    virtual Tensor<1, dim> value(const Point<dim> &p) const override;
    //an array of x and y where y = f(x)
    virtual void value_list(const std::vector<Point<dim>> &p,
                            std::vector<Tensor<1, dim>> &value) const override;
  };


  template <int dim>
  Tensor<1, dim> RightHandSide<dim>::value(const Point<dim> & p) const
  {
    Tensor<1, dim> rhs_value;

    // Compute the right-hand side values based on the given point 'p'
    
        // Example: Compute the right-hand side values as the value of some function f,
        // value = f(x) at x = 'p'
        rhs_value[0] = M_PI*std::cos(M_PI*p[0])*std::cos(2*M_PI*p[1])+M_PI*M_PI*std::cos(M_PI*p[1]);
        rhs_value[1] = 4*M_PI*M_PI*std::sin(2*M_PI*p[0])-2*M_PI*std::sin(M_PI*p[0])*std::sin(2*M_PI*p[1]);
    

    return rhs_value;
  }


//the list of values for y=f(x) for an array of input x using function
// value defined above
  template <int dim>
  void RightHandSide<dim>::value_list(const std::vector<Point<dim>> &vp,
                                      std::vector<Tensor<1, dim>> &values) const
  {
    for (unsigned int c = 0; c < vp.size(); ++c)
      {
        values[c] = RightHandSide<dim>::value(vp[c]);
      }
  }

template <int dim>
  class RightHandSideG : public Function<dim>
    {
    public:
      RightHandSideG()
        : Function<dim>(1)
      {}

      virtual double value(const Point<dim> & p,
                           const unsigned int component = 0) const override;
};

template <int dim>
    double RightHandSideG<dim>::value(const Point<dim> & p,
                                     const unsigned int /*component*/) const
    // Compute the right-hand side values based on the given point 'p'
    
        // Example: Compute the right-hand side values as the value of some function f,
        // value = f(x) at x = 'p'
    {
      double rhs_value;
      rhs_value = -2.0;
      return rhs_value;
    }


  template <int dim>
    class ExactSolution : public Function<dim>
  {
  public:
    ExactSolution()
      : Function<dim>(dim + 1)
    {}

      virtual void vector_value(const Point<dim> &p,
                                Vector<double> &  value) const override;
      virtual void vector_gradient(	const Point< dim > & 	p,
                    std::vector< Tensor< 1, dim> > & 	gradients)	const override;
  };



  template <int dim>
    void ExactSolution<dim>::vector_value(const Point<dim> &p,
                                          Vector<double> &  values) const
    {
      AssertDimension(values.size(), dim + 1);

      values(0) = -3*p[0]+std::cos(M_PI*p[1]);
      values(1) = p[1]+std::sin(2*M_PI*p[0]);
      values(2) = std::sin(M_PI*p[0])*std::cos(2*M_PI*p[1]);
    }
  
  template <int dim>
    void ExactSolution<dim>::vector_gradient(const Point< dim > & 	p,
                                      std::vector< Tensor< 1, dim> > & 	gradients) const
    {
      gradients[0][0] = -3;
      gradients[0][1] = -M_PI*std::sin(M_PI*p[1]);
      gradients[1][0] = 2*M_PI*std::cos(2*M_PI*p[0]);
      gradients[1][1] = 1.0;
      gradients[2][0] = M_PI*std::cos(M_PI*p[0])*std::cos(2*M_PI*p[1]);
      gradients[2][1] = -2*M_PI*std::sin(M_PI*p[0])*std::sin(2*M_PI*p[1]);
    }

  // template <int dim>
  //   class StressTensor_Exact : public TensorFunction<2, dim>
  // {
  // public:
  //   StressTensor_Exact()
  //     : TensorFunction<2, dim>()
  //   {}
  //   //for F
  //   //function to explain the relation y = f(x)
  //   virtual Tensor<2, dim> value(const Point<dim> &p) const override;
  // };

  // template <int dim>
  //   Tensor<2,dim> StressTensor_Exact<dim>::value(const Point<dim> &p) const
  // { Tensor<2, dim> return_value;
  //   {
  //   return_value [0][0] = -std::cos(p[0]*p[1])+2*std::cos(p[0]);
  //   return_value [0][1] = 0;
  //   return_value [1][0] = 0;
  //   return_value [1][1] = -std::cos(p[0]*p[1])+2*std::cos(p[1]);
  //   }

  //   return return_value;
  // }
} // namespace dd_stokes

#endif // ELASTICITY_MFEDD_DATA_H
