# Trabalho 2 - Heuristicas e Metaheuristicas

Implementacoes em C++ dos algoritmos Genetico (AG) e Colonia de Formigas (ACF) para o Caixeiro Viajante (TSP) e a Mochila 0/1.


Cada programa utiliza duas instancias, duas configuracoes de parametros e realiza 10 execucoes por configuracao. As saidas apresentam a funcao objetivo e o tempo de cada execucao, alem do melhor, pior, media e tempo total.

## Estrutura

- `Genetico/Genetico_tsp.cpp`: AG para o TSP (crossover OX e mutacao por troca).
- `Genetico/Genetico_mochila.cpp`: AG para a Mochila 0/1.
- `Formiga/Colonia_tsp.cpp`: ACF para o TSP.
- `Formiga/Colonia_mochila.cpp`: ACF para a Mochila 0/1 (opcional).

Os arquivos `tsp_5`, `tsp_51`, `mochila_4_20` e `mochila_100_1000_1` sao as instancias utilizadas. Os arquivos `resultados_*.txt` guardam as saidas das execucoes realizadas.