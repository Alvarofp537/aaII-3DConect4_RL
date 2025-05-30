# Entrenamiento agentes para Conecta 4 3D de 4 jugadores

Hemos planteado el problema con Aprendizaje por Refuerzo, comparando los resultados entre 3 arquitecturas diferentes: Actor to Critic (A2C), Dueling Deep Q Learning (DQN) y Proximal Policy Optimization (PPO).

Para entrenarlos, primero hemos dejado que entrenen entre sí, para que descubran cómo va el juego, y luego hemos hecho que aprendan de unos `profesores` heurísticos, con posibles modos de juego.

## Índice
  - [Entrega](#entrega)
  - [Estrategia](#estrategia)
  - [Entrenamiento](#entrenamiento)
    - [Cuestiones generales](#cuestiones-generales)
    - [Modelos profesores](#modelos-profesores)
    - [A2C](#a2c)
    - [DQN](#dqn)
    - [PPO](#ppo)


## Entrega
Entregamos como mejores modelos basados en redes neuronales:
- AbderramanIII (PPO)
- AMAgno (DQN)
> [!NOTE]
> Ambos cargan sus respectivos aprendizajes de los archivos adjuntados con el mismo nombre.

y 2 heurísticos que ganaron muchas partidas:
- Almanzor (IntelligentSelfish)
- Bismark (Minimax)

## Estrategia
Lo primero que hicimos fue pensar que probablemente el resto de la clase iba a intentar engañarnos en el entrenamiento conjunto, por lo que hemos basado en este axioma nuestra decisión de votar siempre 0 rondas de entrenamiento y, en caso de haber entrenamiento, hacer que el modelo juegue de manera aleatoria.

## Entrenamiento

### Cuestiones generales
Antes de empezar a entrenar, pensamos que, para que el modelo pudiese aprender bien, tendría que poder diferenciar sus fichas de las demás. Para eso, convertimos el tablero antes de seleccionar acción de la siguiente manera:
> Cambiamos la posición del modelo a -1 y luego, hacemos que los oponentes sean siempre 1, 2, 3.

Con eso conseguimos que, independientemente de la posición en la que juegue, sepa reconocer quién es y el número de tableros que tenga que aprender sea menor.

Al empezar a entrenar y ver que no aprendía, nos fijamos en que las recompensas eran muy escasas y además se aplicaban solo al ganar o realizar una acción inválida. Esto es un problema porque el `learn` se llama en cada acción, y las acciones que realmente te hacen ganar no se premian.

Por lo tanto, definimos una función que calcula una recompensa por cada tablero, basándonos en el número de posibilidades de ganar y cómo de cerca estábamos de ello, menos la probabilidad de ganar de los contrincantes. Así, si consigues mejorar el tablero, la recompensa es inmediata.

Después de entrenar un largo rato, descubrimos también que, principalmente contra ciertos profesores en específico, pero en general también, los modelos se encabezonaban con una posición, incluso cuando la fila estaba llena. El modelo no conseguía aprender pese a las recompensas muy negativas, por lo que tuvimos que aplicar una máscara que pone a 0 la probabilidad o valor Q de las posiciones inválidas para esa situación.

### Modelos profesores
Creamos 3 modelos heurísticos principales para entrenar, que luego modificamos y mezclamos.

1. **Selfish**:  
   Este modelo lo único que hace es, sabiendo que hay 4 jugadores, confiar en que otro se encargue de bloquear. Entonces busca su línea más larga y coloca otra ficha en ella.  
   Con este modelo el agente aprende a bloquear al oponente.

2. **Blocker**:  
   Este modelo es el opuesto del anterior. Se encarga de bloquear siempre que ve que alguien tiene 3 en raya y, por consecuencia, va a ganar.  
   Con este modelo el agente aprende a evitar que le bloqueen (abrir las líneas por los dos lados, ...).

3. **Minimax**:  
   Este modelo se ve como modelo inteligente y se ha usado para refinar el entrenamiento al final.  
   Al ser bastante lento, no han podido entrenar con él muchas iteraciones.

Luego hemos incluido modificaciones de los anteriores:

4. **Intelligent Selfish**:  
    Este modelo funciona igual que Selfish, pero con la única diferencia de que si la ficha que va a poner le da la victoria a otro oponente, no la pone.

5. **Blocker Deluxe**:  
    Modelo análogo al Blocker, con la diferencia de que busca la línea más larga, sin importar que no tengan 3, y la bloquea.

### A2C
Este fue el primer modelo que implementamos y el que peor ha aprendido. No conseguimos que se adecuase al problema.  
Probamos con diferentes arquitecturas de la red, primero aplanando el tablero, como en el resto de los modelos, pero también probamos con una convolucional 3D para ver si aprendía mejor el tablero.
También intentamos que aprendiese acción a acción y luego con partidas completas.

### DQN
Fue el segundo modelo que implementamos y le costó aprender, pero acabó aprendiendo de todos, aunque sin llegar a ganar al Minimax y perdiendo, dependiendo de la partida, con IntelligentSelfish.  
Aun así, es un modelo que ha aprendido bastante y que, dependiendo de la combinación de agentes que haya en la partida, es el mejor modelo que tenemos.
Para DQN también intentamos que aprendiese primero acción a acción y luego por bloque de acciones. Este segundo enfoque permite mejorar las recompensas, y darlas acumuladas. Además hicimos que el modelo para calcular recompensas futuras en la función de bellman se actualizase con menos frecuencia, añadiendo estabilidad al aprendizaje. Estas mejoras ayudaron a que aprendiese más. 

### PPO
Por último entrenamos un PPO, y fue el modelo que más rápido aprendió. Probablemente también porque todo estaba ya muy preparado por la experiencia con los anteriores y fue solo ejecutarlo.  
Es el modelo que mejor funciona en la mayoría de los casos, llegando a ganar a Minimax con una frecuencia relativamente buena.
PPO aprende únicamente al acabar la partida y es el modelo que mejor ha aprendido.