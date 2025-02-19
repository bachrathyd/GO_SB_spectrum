# Example usage:
for order in 1:8
    #coeffs = linear_multistep_coeffs(order)
    #betas = LinearMultiStepCoeff(order, BigFloat)
    betas = LinearMultiStepCoeff(order,BigFloat)
    #@show error = norm(coeffs .- betas)
       println("Adams–Bashforth coefficients for order $order: ", betas)
    #println("Adams–Bashforth coefficients errorfor order $error: ")
end