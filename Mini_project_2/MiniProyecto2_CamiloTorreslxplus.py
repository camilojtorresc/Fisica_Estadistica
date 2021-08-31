import numpy as np
import matplotlib.pyplot as plt
import timeit

# --------------------------- Calculo de los microestados -------------------------------------

# Funcion que agrega, de uno en uno, el numero de spines del sistema
def spin_throw(spinThrow_n1): # spinThrow_n1 es la lista para los spines
    spin = [-1, 1]
    # Si la primera lista esta vacia, se llena con los valores de 1 y -1
    if (np.array(spinThrow_n1).size==0):
        spinThrow_n2=np.array([[i] for i in spin])
        
    # Si la primera lista no esta vacia
    else:
        spinThrow_n2 = [] # La lista spinThrow_n2 se vacia
        for d in spinThrow_n1: # Se recorre cada elemento de spinThrow_n1
            for t in spin: 
                spinThrow_n2.append([*d,t]) # A cada elemento de spinThrow_n1 se le agrega un -1 y luego un 1 y
                                            # se agrega a una nueva lista.
    return spinThrow_n2

# Funcion que crea una matriz con todas las combinaciones de espines del sistema
def microstates(N):
    spinThrow = []
    print(" ")
    print("Creando los microestados...")
    for i in range(N):
        print("Spin",i+1,"added.")
        spinThrow = spin_throw(spinThrow)
    print("Microestados creados.")
    print(" ")
    return np.array(spinThrow)

# --------------------------- Calculo de la densidad de energias -------------------------------------

# Para las condiciones de frontera: 
# Funcion que devuelve, para el microestado i-esimo, las posiciones y cuales son los de microestados con condiciones de frontera periodicas para este 
def boundary_cond(i, L, microstates):
    N=L*L
    return (microstates[(i//L)*L+(i+1)%L], microstates[(i+L)%N],
            microstates[(i//L)*L+(i-1)%L], microstates[(i-L)%N])


# La energia es el negativo de la sumatoria del producto de cada microestado con los espines primeros vecinos
def energia(L, microstates):
    return -0.5*sum(microstates[i]*j
                    for i in range(L*L)
                    for j in boundary_cond(i, L, microstates)
                   )


def densidadenergia(L, microstates):
    lista_micro = []
    den_energias = [] 
    funcion_particion =[]
    dic_energias={}
    
    for estado in microstates:
        
        energia_i = energia(L,estado) # Se calcula la energia de cada microestado

        if energia_i not in dic_energias: # Se realiza el conteo de energia y determinar la degenerancia
            dic_energias[energia_i] = 0
        dic_energias[energia_i] += 1

        lista_micro.append(estado)
        den_energias.append(energia_i)

    en_items = list(dic_energias.items())
    
    energs,rept=np.array([a[0] for a in en_items]), np.array([a[1] for a in en_items])
    
    indx=np.argsort(energs)       # Para organizar
    energias_final=energs[indx]   #
    degeneracion=rept[indx]

    return energias_final, degeneracion, lista_micro, den_energias


# ------------------------------ Calcular la funcion particion ---------------------------------

# Para calcular la funcion particion.
# Con la suma sobre las energias y sobre los microestados.
def funcionParticion(Temperature,degeneracion,energias,lista_micro, den_energias):
    k = 1
    funcpart_energ = []
    funcpart_micro = []
    
    for temp in Temperature:
        func_energ = sum(degeneracion*np.exp(-energias/(k*temp))) #Para k=1, con una temperatura dada, hace la suma sobre todas las energias
        func_micro = 0
        
        for i in range(len(lista_micro)):   # Para k=1, con una temperatura dada, hace la suma sobre todos los microestados
            E = den_energias[i]  # Energia de cada microestado            
            func_micro = func_micro + np.exp(-E/(k*temp))

        funcpart_energ.append(func_energ)
        funcpart_micro.append(func_micro)

    return funcpart_energ, funcpart_micro

# ------------------------------ Calcular la energia media  ---------------------------------

def energiamedia(Temperature, energias, degeneracion,Z):    
    E_prom  = [] # Energia media
    E2_prom = [] # Energia cuadrada media
    sumaEner, sumaEner2 = 0,0
    i=0
    for temp in Temperature:
        sumaEner = sum(energias*degeneracion*np.exp(-energias/temp)) # Se hace -dZ/dB
        e_prom = sumaEner/Z[i]                                          # Se divide entre la funcion particion 
        
        sumaEner2 = sum(energias**2*degeneracion*np.exp(-energias/temp))
        e2_prom = sumaEner2/Z[i] 
        
        #print("sum",sumaEner2)
        #print("Z",Z[i])
        E_prom.append(e_prom)
        #print("Epro",e_prom)
        E2_prom.append(e2_prom)
        #print(e2_prom)
        i+=1
    
    E_prom = np.array(E_prom)
    E2_prom = np.array(E2_prom)
        
    return E_prom, E2_prom

# ------------------------------ Calcular el calor especifico  ---------------------------------
def calorespecifico(N,Temperatures,EnergiaMedia):         
    CalorEsp=[]
    jj = 0

    for t in range(len(EnergiaMedia[1])):
        c_v = (EnergiaMedia[1][t]-(EnergiaMedia[0][t])**2)/(N*(Temperatures[t]**2)) # Se calcula el calor especifico para una lista
        CalorEsp.append(c_v)                                        # de energias medias previamente calculadas con 
                                                                    # un linspace de temperaturas, que tambien se 
    return CalorEsp                                                 # debe usar en esta funcion.


# ----------------------------- Implementacion de las funciones --------------------------------

Temp = np.linspace(0.1,30.0,300)
timer =  []
desvest = []

# **********************************************************************************************
# Para L = 2, N = 2x2

print("Para L=1, N=1x1=1")

start2 = timeit.default_timer()

L2 = 2
N2=L2*L2
microestados2 = microstates(N2)
energia2, degeneracion2, lista_micro2,den_energias2 = densidadenergia(L2, microestados2)

print(" Microestados | Energia")
for i in range(10):
    print (lista_micro2[i], "| {:>5}".format(den_energias2[i]),end=' ')
    print()
print()

for i in range(len(energia2)):
    print ( "| {:>5}".format(energia2[i]),  "| {:>5}".format(degeneracion2[i]))
print()

standardeviation2 = np.std(den_energias2)
print("La desviación estandar para N=4 es: ",standardeviation2,"\n")
desvest.append(standardeviation2)

fig1, ax1 = plt.subplots(figsize=(10,10))
ax1.bar(energia2, degeneracion2, label='data',edgecolor = 'black',color='b')
ax1.set_xlabel("Energy", fontsize=20)
ax1.set_ylabel("Number of micro-states with the same energy", fontsize=16)
ax1.set_title("Ising model for a square of 4 spins", fontsize=18)
#ax1.xticks(energia2)
ax1.grid()
fig1.savefig("plots/Histogram_N4spins.png")

# Calcular la energia media para N=2x2
funcpart2 = funcionParticion(Temp, degeneracion2, energia2, lista_micro2, den_energias2)
EnergiaMedia2 = energiamedia(Temp, energia2, degeneracion2, funcpart2[0])
cv2 = calorespecifico(N2,Temp, EnergiaMedia2)

stop2 = timeit.default_timer()

time2 = stop2-start2
timer.append(time2)
print("Time N = 4, ", time2)
print("\n")
# **********************************************************************************************
# Para L = 3, N = 3x3

print("Para L=3, N=3x3=9")

start3 = timeit.default_timer()

L3 = 3
N3=L3*L3
microestados3 = microstates(N3)
energia3, degeneracion3, lista_micro3,den_energias3 = densidadenergia(L3, microestados3)

print(" Microestados | Energia")
for i in range(10):
    print (lista_micro3[i], "| {:>5}".format(den_energias3[i]),end=' ')
    print()
print()

for i in range(len(energia3)):
    print ( "| {:>5}".format(energia3[i]),  "| {:>5}".format(degeneracion3[i]))
print()
standardeviation3 = np.std(den_energias3)
print("La desviación estandar para N=9 es: ",standardeviation3,"\n")
desvest.append(standardeviation3)

fig2, ax2 = plt.subplots(figsize=(10,10))
ax2.bar(energia3, degeneracion3, label='data',edgecolor = 'black',color='b')
ax2.set_xlabel("Energy", fontsize=20)
ax2.set_ylabel("Number of micro-states with the same energy", fontsize=16)
ax2.set_title("Ising model for a square of 9 spins", fontsize=18)
#ax2.xticks(energia3)
ax2.grid()
fig2.savefig("plots/Histogram_N9spins.png")

# Calcular la energia media para N=3x3
funcpart3 = funcionParticion(Temp, degeneracion3, energia3, lista_micro3, den_energias3)
EnergiaMedia3 = energiamedia(Temp, energia3, degeneracion3, funcpart3[0])
cv3 = calorespecifico(N3,Temp, EnergiaMedia3)

stop3 = timeit.default_timer()

time3 = stop3-start3
timer.append(time3)
print("Time N = 9, ", time3)
print("\n")
# **********************************************************************************************

# **********************************************************************************************
# Para L = 4, N = 4x4

print("Para L=4, N=4x4=16")

start4 = timeit.default_timer()

L4 = 4
N4=L4*L4
microestados4 = microstates(N4)
energia4, degeneracion4, lista_micro4,den_energias4 = densidadenergia(L4, microestados4)

print(" Microestados | Energia")
for i in range(10):
    print (lista_micro4[i], "| {:>5}".format(den_energias4[i]),end=' ')
    print()
print()

for i in range(len(energia4)):
    print ( "| {:>5}".format(energia4[i]),  "| {:>5}".format(degeneracion4[i]))
print()

standardeviation4 = np.std(den_energias4)
print("La desviación estandar para N=16 es: ",standardeviation4,"\n")
desvest.append(standardeviation4)

fig3, ax3 = plt.subplots(figsize=(10,10))
ax3.bar(energia4, degeneracion4, label='data',edgecolor = 'black',color='b')
ax3.set_xlabel("Energy", fontsize=20)
ax3.set_ylabel("Number of micro-states with the same energy", fontsize=16)
ax3.set_title("Ising model for a square of 16 spins", fontsize=18)
#ax2.xticks(energia3)
ax3.grid()
fig3.savefig("plots/Histogram_N16spins.png")

# Calcular la energia media para N=3x3
funcpart4 = funcionParticion(Temp, degeneracion4, energia4, lista_micro4, den_energias4)
EnergiaMedia4 = energiamedia(Temp, energia4, degeneracion4, funcpart4[0])
cv4 = calorespecifico(N4,Temp, EnergiaMedia4)

stop4 = timeit.default_timer()

time4 = stop4-start4
timer.append(time4)
print("Time N = 16, ", time4)
print("\n")
# **********************************************************************************************

# **********************************************************************************************
# Para L = 5, N = 5x5

print("Para L=5, N=5x5=25")

start5 = timeit.default_timer()

L5 = 5
N5=L5*L5
microestados5 = microstates(N5)
energia5, degeneracion5, lista_micro5,den_energias5 = densidadenergia(L5, microestados5)

print(" Microestados | Energia")
for i in range(10):
    print (lista_micro5[i], "| {:>5}".format(den_energias5[i]),end=' ')
    print()
print()

for i in range(len(energia4)):
    print ( "| {:>5}".format(energia5[i]),  "| {:>5}".format(degeneracion5[i]))
print()

standardeviation5 = np.std(den_energias5)
print("La desviación estandar para N=5 es: ",standardeviation5,"\n")
desvest.append(standardeviation5)

fig4, ax4 = plt.subplots(figsize=(10,10))
ax4.bar(energia5, degeneracion5, label='data',edgecolor = 'black',color='b')
ax4.set_xlabel("Energy", fontsize=20)
ax4.set_ylabel("Number of micro-states with the same energy", fontsize=16)
ax4.set_title("Ising model for a square of 16 spins", fontsize=18)
#ax2.xticks(energia3)
ax4.grid()
fig4.savefig("plots/Histogram_N25spins.png")

# Calcular la energia media para N=3x3
funcpart5 = funcionParticion(Temp, degeneracion5, energia5, lista_micro5, den_energias5)
EnergiaMedia5 = energiamedia(Temp, energia5, degeneracion5, funcpart5[0])
cv5 = calorespecifico(N5,Temp, EnergiaMedia5)

stop5 = timeit.default_timer()

time5 = stop5-start5
timer.append(time5)
print("Time N = 25, ", time5)
print("\n")
# **********************************************************************************************
# **********************************************************************************************
# Para L = 6, N = 6x6

print("Para L=6, N=6x6=36")

start6 = timeit.default_timer()

L6 = 6
N6=L6*L6
microestados6 = microstates(N6)
energia6, degeneracion6, lista_micro6,den_energias6 = densidadenergia(L6, microestados6)

print(" Microestados | Energia")
for i in range(10):
    print (lista_micro6[i], "| {:>5}".format(den_energias6[i]),end=' ')
    print()
print()

for i in range(len(energia6)):
    print ( "| {:>5}".format(energia6[i]),  "| {:>5}".format(degeneracion6[i]))
print()

standardeviation6 = np.std(den_energias6)
print("La desviación estandar para N=36 es: ",standardeviation6,"\n")
desvest.append(standardeviation6)

fig5, ax5 = plt.subplots(figsize=(10,10))
ax5.bar(energia6, degeneracion6, label='data',edgecolor = 'black',color='b')
ax6.set_xlabel("Energy", fontsize=20)
ax6.set_ylabel("Number of micro-states with the same energy", fontsize=16)
ax5.set_title("Ising model for a square of 36 spins", fontsize=18)
#ax2.xticks(energia3)
ax5.grid()
fig5.savefig("plots/Histogram_N36spins.png")

# Calcular la energia media para N=6x6
funcpart6 = funcionParticion(Temp, degeneracion6, energia6, lista_micro6, den_energias6)
EnergiaMedia6 = energiamedia(Temp, energia6, degeneracion6, funcpart6[0])
cv6 = calorespecifico(N6,Temp, EnergiaMedia6)

stop6 = timeit.default_timer()

time6 = stop6-start6
timer.append(time6)
print("Time N = 36, ", time6)
print("\n")
# **********************************************************************************************


print("tiempos de computo:")
print(timer,"\n")

print("Desviacion estandar:")
print(desvest,"\n")
# Crea y guarda la figura con la energia media

fig11, ax11 = plt.subplots(figsize=(10,5))
ax11.plot(Temp,EnergiaMedia6[0],"y",label="N=6x6")
ax11.plot(Temp,EnergiaMedia5[0],"g",label="N=5x5")
ax11.plot(Temp,EnergiaMedia4[0],"r",label="N=4x4")
ax11.plot(Temp,EnergiaMedia3[0],"m",label="N=3x3")
ax11.plot(Temp,EnergiaMedia2[0],"b",label="N=2x2")
ax11.set_xlabel("Temperature T(K)", fontsize=20)
ax11.set_ylabel("Average energy <E>", fontsize=20)
ax11.set_title("Average energy in function of temperature", fontsize=20)
ax11.legend()
ax11.grid()
fig11.savefig("plots/AvarageEnergy.png")

# Crea y guarda la figura con los calores especificos
fig22, ax22 = plt.subplots(figsize=(10,5))
ax22.plot(Temp,cv6,"g",label="N=6x6")
ax22.plot(Temp,cv5,"g",label="N=5x5")
ax22.plot(Temp,cv4,"b",label="N=4x4")
ax22.plot(Temp,cv3,"r",label="N=3x3")
ax22.plot(Temp,cv2,"m",label="N=2x2")
ax22.set_xlabel("Temperature $T(K)$", fontsize=20)
ax22.set_ylabel("Specific heat capacity $c_{v}$", fontsize=20)
ax22.set_title("Specific heat capacity in function of Temperature", fontsize=16)
ax22.legend()
ax22.set_xlim(0,10)
ax22.grid()
fig22.savefig("plots/SpecificHeat.png")


