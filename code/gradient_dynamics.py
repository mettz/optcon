# Computation of the gradient of the discretized dynamics equations
# nabla0 = np.array( # Derivate of x_dot with respect to (x,y,psi,V,beta,psi_dot)
#     [
#         1,
#         0,
#         -dt * V * (np.cos(beta) * np.sin(psi) + np.sin(beta) * np.cos(psi)),
#         dt * (np.cos(beta) * np.cos(psi) - np.sin(beta) * np.sin(psi)),
#         -dt * V * (np.sin(beta) * np.cos(psi) - np.cos(beta) * np.sin(psi)),
#         0,
#     ] #OK
# )
# nabla1 = np.array( # Derivate of y_dot with respect to (x,y,psi,V,beta,psi_dot)
#     [
#         0,
#         1,
#         dt * V * (np.cos(beta) * np.cos(psi) - np.sin(beta) * np.sin(psi)),
#         dt * (np.cos(beta) * np.sin(psi) + np.sin(beta) * np.cos(psi)),
#         -dt * V * (np.sin(beta) * np.sin(psi) - V * np.cos(beta) * np.cos(psi)),
#         0,
#     ] #OK
# )
# nabla2 = np.array([0, 0, 1, 0, 0, dt]) # Derivate of psi_dot with respect to (x,y,psi,V,beta,psi_dot) #OK
# nabla3 = np.array( # Derivate of V_dot with respect to (x,y,psi,V,beta,psi_dot)
#     [
#         0,
#         0,
#         0,
#         (mass*V**2*np.cos(beta) + (-psi_dot * a * np.sin(delta-beta) * Fzf - Fzr * psi_dot * b) * dt * mu) / (np.cos(beta) * mass * V**2),
#         dt * (1/mass) * (Fzf * mu * (-(np.sin(beta) * (V * np.sin(beta) + psi_dot * a) * np.sin(beta-delta) / (V * np.cos(beta)**2) - np.sin(beta-delta) - (V * np.sin(beta) + psi_dot * a) * np.cos(beta-delta))/ (V * np.cos(beta))) - Fx * np.sin(beta-delta) + (Fzr * mu * np.sin(beta) * (psi_dot * b - V * np.sin(beta))) / (V * np.cos(beta)**2) - Fzr *mu) ,
#         ((a * np.sin(delta-beta) * Fzf + Fzr * b) * dt * mu) / (mass * V * np.cos(beta)),
#     ] #A Luca sembra ok, magari controllare
# )
# nabla4 = np.array( # Derivate of beta_dot with respect to (x,y,psi,V,beta,psi_dot)
#     [ #CONTROLLARE
#         0,
#         0,
#         0,
#         (-np.sin(delta-beta)*Fx+((delta-np.tan(beta)*np.cos(delta-beta)*Fzf-np.tan(beta)*Fzr)*mu)*V+(2*beta*(1/np.cos(beta))*psi_dot*Fzr-2*a*(1/np.cos(beta))*np.cos(delta-beta)*Fzf*psi_dot)*mu)/mass*V**3,
#         -(((V*np.sin(delta)*Fx+(V*np.sin(delta)+V*delta*np.cos(delta))*Fzf*mu)*np.cos(beta)**-psi*Fzr*mu*beta+V*np.sin(delta)*Fzf*mu)*np.sin(beta))+(V*np.cos(delta)*Fx),
#         -1,
#     ]
# )
# nabla5 = np.array( # Derivate of psi_dot_dot with respect to (x,y,psi,V,beta,psi_dot)
#     [ #MANCA LA DERIVATA
#         0,
#         0,
#         0,
#         psi_dot * a * (np.cos(beta) * a + 1) * b  ,
#         0,
#         1
#      ]
#     )

# Code for debugging and testing
# print('nabla0', nabla0.shape)
# prova = np.array([nabla0, nabla1, nabla2, nabla3, nabla4, nabla5])

# print('prova.shape', prova.shape)
# print('prova', prova)

# fx = np.column_stack((nabla0, nabla1, nabla2, nabla3, nabla4, nabla5))
# print('fx', fx.shape)
# print('fx', fx)

# fu = np.array(
#     [
#         [ # Derivative of x_dot,y_dot,psi_dot,V_dot,beta_dot,psi_dot_dot with respect to delta
#             0,
#             0,
#             0,
#             dt * (1 / mass) * (Fx * np.sin(beta - delta) - Fyf * np.cos(beta - delta)),
#             dt * (1 / mass * V) * (Fyf * np.sin(beta - delta) + Fx * np.cos(beta - delta)),
#             dt * 1 / Iz * (Fx * np.cos(delta) - Fyf * np.sin(delta)) * a,
#         ],
#         [ # Derivative of x_dot,y_dot,psi_dot,V_dot,beta_dot,psi_dot_dot with respect to Fx
#             0,
#             0,
#             0,
#             dt * 1 / mass * np.cos(beta - delta),
#             dt * (1 / (mass * V) * (-np.sin(beta - delta))),
#             dt * 1 / Iz * np.sin(delta) * a,
#         ],
#     ]
# )
