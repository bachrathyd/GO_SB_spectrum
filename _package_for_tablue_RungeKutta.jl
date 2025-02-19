using RungeKutta
BT = TableauGauss(Float64, 10)
BT = TableauRalston3(Float16)
B=ExplicitMidpoint(Float64)

#ExplicitEuler/ForwardEuler, ExplicitMidpoint, Heun2, Heun3, Ralston2, Ralston3, Runge2, Kutta3, 
#RK21, RK22, RK31, RK32, RK4, RK41, RK42, RK416, RK438, RK5, SSPRK2, SSPRK3