#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <random>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <cmath>
#include <limits>

using namespace std;

struct Cidade {
    int id;
    int x;
    int y;
};

// Funcao reaproveitada para limpeza e leitura do arquivo.
vector<int> extrairDadosLimpos(const string& nomeArquivo) {
    vector<int> dados;
    ifstream arquivo(nomeArquivo);
    string linha;
    if (!arquivo.is_open()) return dados;

    while (getline(arquivo, linha)) {
        size_t pos = linha.find('#');
        if (pos != string::npos) linha = linha.substr(0, pos);

        stringstream ss(linha);
        int valor;
        while (ss >> valor) dados.push_back(valor);
    }
    arquivo.close();
    return dados;
}

// Funcao objetivo do TSP: soma das distancias, incluindo o retorno ao inicio.
int avaliar_tsp(const vector<int>& caminho, const vector<vector<int>>& matriz) {
    int custo = 0, n = (int)caminho.size();
    if (n <= 1) return 0;
    for (int i = 0; i < n - 1; i++) {
        custo += matriz[caminho[i]][caminho[i + 1]];
    }
    custo += matriz[caminho[n - 1]][caminho[0]];
    return custo;
}

struct Individuo {
    vector<int> cromossomo;
    int custo = numeric_limits<int>::max();
};

struct ParametrosAG {
    int tamanho_populacao;
    int max_geracoes;
    double taxa_reproducao;
    double prob_crossover;
    double prob_mutacao;
    int config_id;
};

struct ResultadoExecucaoTSP {
    int execucao;
    int custo;
    double tempo_ms;
};

// Avalia uma rota que representa uma permutacao das cidades.
Individuo criar_individuo(vector<int> cromossomo, const vector<vector<int>>& matriz) {
    Individuo ind;
    ind.cromossomo = move(cromossomo);
    ind.custo = avaliar_tsp(ind.cromossomo, matriz);
    return ind;
}

// Populacao inicial aleatoria com rotas completas sem cidades repetidas.
vector<Individuo> gerar_populacao(const vector<vector<int>>& matriz,
                                  int tamanho, mt19937& gen) {
    vector<Individuo> populacao;
    populacao.reserve(tamanho);
    vector<int> rota(matriz.size());
    iota(rota.begin(), rota.end(), 0);

    for (int i = 0; i < tamanho; i++) {
        shuffle(rota.begin(), rota.end(), gen);
        populacao.push_back(criar_individuo(rota, matriz));
    }
    return populacao;
}

// Roleta para minimizacao: o menor custo recebe maior chance de ser pai.
// O indice excluido permite selecionar dois pais distintos.
int selecionar_roleta(const vector<Individuo>& populacao, mt19937& gen,
                     int excluido = -1) {
    double soma_fitness = 0.0;
    for (int i = 0; i < (int)populacao.size(); i++) {
        if (i != excluido) soma_fitness += 1.0 / max(1, populacao[i].custo);
    }

    if (!(soma_fitness > 0.0)) {
        vector<int> candidatos;
        for (int i = 0; i < (int)populacao.size(); i++) {
            if (i != excluido) candidatos.push_back(i);
        }
        uniform_int_distribution<int> sorteio(0, (int)candidatos.size() - 1);
        return candidatos[sorteio(gen)];
    }

    uniform_real_distribution<double> sorteio(0.0, soma_fitness);
    double alvo = sorteio(gen);
    double acumulado = 0.0;
    int ultimo_valido = -1;

    for (int i = 0; i < (int)populacao.size(); i++) {
        if (i == excluido) continue;
        ultimo_valido = i;
        acumulado += 1.0 / max(1, populacao[i].custo);
        if (alvo <= acumulado) return i;
    }
    return ultimo_valido;
}

// Order Crossover (OX): preserva um trecho de um pai e completa com a
// ordem das cidades do outro pai, sem repeticoes.
vector<int> construir_filho_ox(const vector<int>& pai1, const vector<int>& pai2,
                              int inicio, int fim) {
    int n = (int)pai1.size();
    vector<int> filho(n, -1);
    vector<bool> usado(n, false);

    for (int i = inicio; i <= fim; i++) {
        filho[i] = pai1[i];
        usado[pai1[i]] = true;
    }

    int destino = (fim + 1) % n;
    for (int passo = 0; passo < n; passo++) {
        int cidade = pai2[(fim + 1 + passo) % n];
        if (!usado[cidade]) {
            filho[destino] = cidade;
            usado[cidade] = true;
            destino = (destino + 1) % n;
        }
    }
    return filho;
}

// Crossover OX de dois pontos, usado para cromossomos de permutacao.
void aplicar_crossover(vector<int>& filho1, vector<int>& filho2,
                       double probabilidade, mt19937& gen) {
    if (filho1.size() < 2) return;
    bernoulli_distribution fazer_crossover(probabilidade);
    if (!fazer_crossover(gen)) return;

    uniform_int_distribution<int> ponto(0, (int)filho1.size() - 1);
    int inicio = ponto(gen);
    int fim = ponto(gen);
    if (inicio > fim) swap(inicio, fim);

    vector<int> novo1 = construir_filho_ox(filho1, filho2, inicio, fim);
    vector<int> novo2 = construir_filho_ox(filho2, filho1, inicio, fim);
    filho1 = move(novo1);
    filho2 = move(novo2);
}

// Mutacao por troca (swap) de duas cidades, conforme a aula.
void aplicar_mutacao(vector<int>& filho, double probabilidade, mt19937& gen) {
    if (filho.size() < 2) return;
    bernoulli_distribution fazer_mutacao(probabilidade);
    if (fazer_mutacao(gen)) {
        uniform_int_distribution<int> gene(0, (int)filho.size() - 1);
        int pos1 = gene(gen);
        int pos2 = gene(gen);
        while (pos2 == pos1) pos2 = gene(gen);
        swap(filho[pos1], filho[pos2]);
    }
}

// Elitismo: escolhe as rotas de menor custo entre pais e filhos.
vector<Individuo> selecionar_sobreviventes(vector<Individuo>& pais,
                                            vector<Individuo>& filhos, int tamanho) {
    for (Individuo& filho : filhos) pais.push_back(move(filho));

    sort(pais.begin(), pais.end(), [](const Individuo& a, const Individuo& b) {
        return a.custo < b.custo;
    });
    pais.resize(tamanho);
    return move(pais);
}

// Executa o AG: selecao, reproducao, avaliacao e sobrevivencia.
Individuo algoritmo_genetico(const vector<vector<int>>& matriz,
                             const ParametrosAG& p, mt19937& gen) {
    vector<Individuo> populacao = gerar_populacao(matriz, p.tamanho_populacao, gen);

    for (int geracao = 0; geracao < p.max_geracoes; geracao++) {
        int qtd_filhos = (int)round(p.tamanho_populacao * p.taxa_reproducao);
        qtd_filhos = max(2, min(p.tamanho_populacao, qtd_filhos));
        if (qtd_filhos % 2 != 0) qtd_filhos--;

        vector<Individuo> filhos;
        filhos.reserve(qtd_filhos);

        while ((int)filhos.size() < qtd_filhos) {
            int idx1 = selecionar_roleta(populacao, gen);
            int idx2 = selecionar_roleta(populacao, gen, idx1);

            vector<int> filho1 = populacao[idx1].cromossomo;
            vector<int> filho2 = populacao[idx2].cromossomo;

            aplicar_crossover(filho1, filho2, p.prob_crossover, gen);
            aplicar_mutacao(filho1, p.prob_mutacao, gen);
            aplicar_mutacao(filho2, p.prob_mutacao, gen);

            filhos.push_back(criar_individuo(move(filho1), matriz));
            filhos.push_back(criar_individuo(move(filho2), matriz));
        }
        populacao = selecionar_sobreviventes(populacao, filhos, p.tamanho_populacao);
    }
    return populacao.front();
}

void executarGeneticoTSP(const vector<Cidade>& cidades, const vector<vector<int>>& matriz,
                         const ParametrosAG& p) {
    vector<ResultadoExecucaoTSP> resultados(10);
    random_device rd;
    mt19937 gen(rd());
    vector<int> melhor_rota_geral;
    int melhor_custo_geral = numeric_limits<int>::max();

    auto inicio_geral = chrono::steady_clock::now();
    for (int exec = 0; exec < 10; exec++) {
        auto inicio_exec = chrono::steady_clock::now();
        Individuo melhor = algoritmo_genetico(matriz, p, gen);
        auto fim_exec = chrono::steady_clock::now();
        chrono::duration<double, milli> duracao = fim_exec - inicio_exec;

        resultados[exec] = {exec + 1, melhor.custo, duracao.count()};
        if (melhor.custo < melhor_custo_geral) {
            melhor_custo_geral = melhor.custo;
            melhor_rota_geral = melhor.cromossomo;
        }
    }
    auto fim_geral = chrono::steady_clock::now();
    chrono::duration<double, milli> duracao_total = fim_geral - inicio_geral;

    int melhor = resultados[0].custo;
    int pior = resultados[0].custo;
    double soma = 0.0;
    for (const auto& r : resultados) {
        melhor = min(melhor, r.custo);
        pior = max(pior, r.custo);
        soma += r.custo;
    }

    cout << "\n===================================================================\n";
    cout << " TABELA DE RESULTADOS (TSP - GENETICO) - CONFIGURACAO " << p.config_id << "\n";
    cout << " (Pop = " << p.tamanho_populacao << ", Geracoes = " << p.max_geracoes
         << ", Reproducao = " << p.taxa_reproducao << ", Pc = " << p.prob_crossover
         << ", Pm = " << p.prob_mutacao << ")\n";
    cout << "===================================================================\n";
    cout << left << setw(10) << "Execucao" << setw(15) << "Custo (Obj)"
         << setw(15) << "Tempo (ms)" << '\n';
    cout << "-------------------------------------------------------------------\n";
    for (const auto& r : resultados) {
        cout << left << setw(10) << r.execucao << setw(15) << r.custo
             << setw(15) << fixed << setprecision(2) << r.tempo_ms << '\n';
    }
    cout << "-------------------------------------------------------------------\n";
    cout << " ESTATISTICAS FINAIS E TEMPO TOTAL\n";
    cout << "-------------------------------------------------------------------\n";
    cout << left << setw(15) << "Melhor Custo" << setw(15) << "Pior Custo"
         << setw(15) << "Media Custos" << setw(20) << "Tempo Total (ms)" << '\n';
    cout << left << setw(15) << melhor << setw(15) << pior
         << setw(15) << fixed << setprecision(2) << soma / resultados.size()
         << setw(20) << duracao_total.count() << '\n';

    cout << " Melhor rota (ids): ";
    for (int i : melhor_rota_geral) cout << cidades[i].id << " -> ";
    if (!melhor_rota_geral.empty()) cout << cidades[melhor_rota_geral.front()].id;
    cout << "\n===================================================================\n";
}

bool carregar_tsp(const string& nomeArquivo, vector<Cidade>& cidades,
                 vector<vector<int>>& matriz) {
    vector<int> dados = extrairDadosLimpos(nomeArquivo);
    if (dados.empty() || dados.size() % 3 != 0) return false;

    int n = (int)dados.size() / 3;
    cidades.resize(n);
    for (int i = 0, index = 0; i < n; i++, index += 3) {
        cidades[i] = {dados[index], dados[index + 1], dados[index + 2]};
    }

    matriz.assign(n, vector<int>(n, 0));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i != j) {
                double dx = (double)cidades[i].x - cidades[j].x;
                double dy = (double)cidades[i].y - cidades[j].y;
                matriz[i][j] = (int)round(sqrt(dx * dx + dy * dy));
            }
        }
    }
    return true;
}

int main() {
    // Duas configuracoes e duas instancias, como nos trabalhos anteriores.
    const ParametrosAG config1 = {40, 150, 0.80, 0.70, 0.05, 1};
    const ParametrosAG config2 = {80, 300, 1.00, 0.90, 0.10, 2};

    const vector<string> arquivos = {"tsp_5", "tsp_51"};
    for (const string& nomeArquivo : arquivos) {
        vector<Cidade> cidades;
        vector<vector<int>> matriz;
        if (!carregar_tsp(nomeArquivo, cidades, matriz)) {
            cerr << "Erro: arquivo nao encontrado ou formato invalido: " << nomeArquivo << '\n';
            return 1;
        }

        cout << "\nINSTANCIA: " << nomeArquivo << " | Cidades: " << cidades.size() << '\n';
        executarGeneticoTSP(cidades, matriz, config1);
        executarGeneticoTSP(cidades, matriz, config2);
    }
    return 0;
}
