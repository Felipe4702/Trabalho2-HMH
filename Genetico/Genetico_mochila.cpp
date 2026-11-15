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

using namespace std;

struct ItemMochila {
    int id;
    int peso;
    int valor;
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

// Avaliacao da mochila: maximizar valor sem ultrapassar a capacidade.
int avaliar_mochila(const vector<int>& solucao, const vector<ItemMochila>& itens,
                    int cap, int& peso_total_out) {
    int peso = 0;
    int valor = 0;
    for (size_t i = 0; i < solucao.size(); i++) {
        if (solucao[i] == 1) {
            peso += itens[i].peso;
            valor += itens[i].valor;
        }
    }
    peso_total_out = peso;
    if (peso > cap) return 0;
    return valor;
}

struct Individuo {
    vector<int> cromossomo;
    int valor = 0;
    int peso = 0;
};

struct ParametrosAG {
    int tamanho_populacao;
    int max_geracoes;
    double taxa_reproducao;
    double prob_crossover;
    double prob_mutacao;
    int config_id;
};

struct ResultadoExecucaoMochila {
    int execucao;
    int valor_obj;
    int peso_total;
    double tempo_ms;
};

// Corrige um cromossomo inviavel e aproveita a capacidade restante.
// A ordem aleatoria preserva a diversidade na populacao.
void reparar_mochila(vector<int>& s, const vector<ItemMochila>& itens,
                     int cap, mt19937& gen) {
    int peso = 0;
    for (size_t i = 0; i < s.size(); i++) {
        if (s[i] == 1) peso += itens[i].peso;
    }

    vector<int> selecionados;
    for (int i = 0; i < (int)s.size(); i++) {
        if (s[i] == 1) selecionados.push_back(i);
    }
    shuffle(selecionados.begin(), selecionados.end(), gen);

    for (int i : selecionados) {
        if (peso <= cap) break;
        s[i] = 0;
        peso -= itens[i].peso;
    }

    vector<int> candidatos;
    for (int i = 0; i < (int)s.size(); i++) {
        if (s[i] == 0) candidatos.push_back(i);
    }
    shuffle(candidatos.begin(), candidatos.end(), gen);

    for (int i : candidatos) {
        if (peso + itens[i].peso <= cap) {
            s[i] = 1;
            peso += itens[i].peso;
        }
    }
}

// Calcula o fitness do cromossomo.
Individuo criar_individuo(vector<int> cromossomo,
                          const vector<ItemMochila>& itens, int cap, mt19937& gen) {
    reparar_mochila(cromossomo, itens, cap, gen);
    Individuo ind;
    ind.cromossomo = move(cromossomo);
    ind.valor = avaliar_mochila(ind.cromossomo, itens, cap, ind.peso);
    return ind;
}

// Populacao inicial aleatoria com cromossomos binarios.
vector<Individuo> gerar_populacao(const vector<ItemMochila>& itens, int cap,
                                  int tamanho, mt19937& gen) {
    vector<Individuo> populacao;
    populacao.reserve(tamanho);
    bernoulli_distribution bit_aleatorio(0.5);

    for (int i = 0; i < tamanho; i++) {
        vector<int> s(itens.size());
        for (int& bit : s) bit = bit_aleatorio(gen) ? 1 : 0;
        populacao.push_back(criar_individuo(move(s), itens, cap, gen));
    }
    return populacao;
}

// Roleta: a chance de cada pai e proporcional ao seu fitness.
// O indice excluido permite selecionar dois pais distintos.
int selecionar_roleta(const vector<Individuo>& populacao, mt19937& gen,
                     int excluido = -1) {
    double soma_fitness = 0.0;
    for (int i = 0; i < (int)populacao.size(); i++) {
        if (i != excluido) soma_fitness += max(0, populacao[i].valor);
    }

    if (soma_fitness <= 0.0) {
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
        if (i == excluido || populacao[i].valor <= 0) continue;
        ultimo_valido = i;
        acumulado += populacao[i].valor;
        if (alvo <= acumulado) return i;
    }
    return ultimo_valido;
}

// Crossover classico de um ponto, conforme a aula.
void aplicar_crossover(vector<int>& filho1, vector<int>& filho2,
                       double probabilidade, mt19937& gen) {
    if (filho1.size() < 2) return;
    bernoulli_distribution fazer_crossover(probabilidade);
    if (!fazer_crossover(gen)) return;

    uniform_int_distribution<int> ponto(1, (int)filho1.size() - 1);
    int corte = ponto(gen);
    for (int i = corte; i < (int)filho1.size(); i++) {
        swap(filho1[i], filho2[i]);
    }
}

// Mutacao de um gene por flip, quando o filho e sorteado.
void aplicar_mutacao(vector<int>& filho, double probabilidade, mt19937& gen) {
    if (filho.empty()) return;
    bernoulli_distribution fazer_mutacao(probabilidade);
    if (fazer_mutacao(gen)) {
        uniform_int_distribution<int> gene(0, (int)filho.size() - 1);
        int pos = gene(gen);
        filho[pos] = 1 - filho[pos];
    }
}

// Elitismo: escolhe os individuos com melhor fitness entre pais e filhos.
vector<Individuo> selecionar_sobreviventes(vector<Individuo>& pais,
                                            vector<Individuo>& filhos, int tamanho) {
    for (Individuo& filho : filhos) pais.push_back(move(filho));

    sort(pais.begin(), pais.end(), [](const Individuo& a, const Individuo& b) {
        if (a.valor != b.valor) return a.valor > b.valor;
        return a.peso < b.peso;
    });
    pais.resize(tamanho);
    return move(pais);
}

// Executa o AG: selecao, reproducao, avaliacao e sobrevivencia.
Individuo algoritmo_genetico(const vector<ItemMochila>& itens, int cap,
                            const ParametrosAG& p, mt19937& gen) {
    vector<Individuo> populacao = gerar_populacao(itens, cap, p.tamanho_populacao, gen);

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

            filhos.push_back(criar_individuo(move(filho1), itens, cap, gen));
            filhos.push_back(criar_individuo(move(filho2), itens, cap, gen));
        }

        populacao = selecionar_sobreviventes(populacao, filhos, p.tamanho_populacao);
    }

    // A populacao final esta ordenada pelo elitismo da ultima geracao.
    return populacao.front();
}

void executarGeneticoMochila(const vector<ItemMochila>& itens, int cap,
                            const ParametrosAG& p) {
    vector<ResultadoExecucaoMochila> resultados(10);
    random_device rd;
    mt19937 gen(rd());
    Individuo melhor_solucao;
    melhor_solucao.valor = -1;

    auto inicio_geral = chrono::steady_clock::now();
    for (int exec = 0; exec < 10; exec++) {
        auto inicio_exec = chrono::steady_clock::now();
        Individuo melhor = algoritmo_genetico(itens, cap, p, gen);
        auto fim_exec = chrono::steady_clock::now();
        chrono::duration<double, milli> duracao = fim_exec - inicio_exec;

        resultados[exec] = {exec + 1, melhor.valor, melhor.peso, duracao.count()};
        if (melhor.valor > melhor_solucao.valor) melhor_solucao = melhor;
    }
    auto fim_geral = chrono::steady_clock::now();
    chrono::duration<double, milli> duracao_total = fim_geral - inicio_geral;

    int melhor_valor = resultados[0].valor_obj;
    int pior_valor = resultados[0].valor_obj;
    double soma = 0.0;
    for (const auto& r : resultados) {
        melhor_valor = max(melhor_valor, r.valor_obj);
        pior_valor = min(pior_valor, r.valor_obj);
        soma += r.valor_obj;
    }

    cout << "\n===================================================================\n";
    cout << " TABELA DE RESULTADOS (MOCHILA - GENETICO) - CONFIGURACAO " << p.config_id << "\n";
    cout << " (Pop = " << p.tamanho_populacao << ", Geracoes = " << p.max_geracoes
         << ", Reproducao = " << p.taxa_reproducao << ", Pc = " << p.prob_crossover
         << ", Pm = " << p.prob_mutacao << ")\n";
    cout << "===================================================================\n";
    cout << left << setw(10) << "Execucao" << setw(15) << "Valor (Obj)"
         << setw(15) << "Peso Total" << setw(15) << "Tempo (ms)" << '\n';
    cout << "-------------------------------------------------------------------\n";
    for (const auto& r : resultados) {
        cout << left << setw(10) << r.execucao << setw(15) << r.valor_obj
             << setw(15) << r.peso_total << setw(15) << fixed << setprecision(2)
             << r.tempo_ms << '\n';
    }
    cout << "-------------------------------------------------------------------\n";
    cout << " ESTATISTICAS FINAIS E TEMPO TOTAL\n";
    cout << "-------------------------------------------------------------------\n";
    cout << left << setw(15) << "Melhor Valor" << setw(15) << "Pior Valor"
         << setw(15) << "Media Valores" << setw(20) << "Tempo Total (ms)" << '\n';
    cout << left << setw(15) << melhor_valor << setw(15) << pior_valor
         << setw(15) << fixed << setprecision(2) << soma / resultados.size()
         << setw(20) << duracao_total.count() << '\n';
    cout << " Itens da melhor solucao: ";
    bool primeiro = true;
    for (int i = 0; i < (int)itens.size(); i++) {
        if (melhor_solucao.cromossomo[i] == 1) {
            if (!primeiro) cout << ", ";
            cout << itens[i].id;
            primeiro = false;
        }
    }
    if (primeiro) cout << "nenhum";
    cout << "\n===================================================================\n";
}

bool carregar_mochila(const string& nomeArquivo, vector<ItemMochila>& itens, int& cap) {
    vector<int> dados = extrairDadosLimpos(nomeArquivo);
    if (dados.size() < 2 || dados[0] <= 0 || dados[1] < 0) return false;

    int numItens = dados[0];
    cap = dados[1];
    if (dados.size() < (size_t)(2 + 2 * numItens)) return false;

    itens.clear();
    for (int i = 0, index = 2; i < numItens; i++, index += 2) {
        int valor = dados[index];
        int peso = dados[index + 1];
        if (peso < 0 || valor < 0) return false;
        itens.push_back({i + 1, peso, valor});
    }
    return true;
}

int main() {
    // Duas configuracoes e duas instancias, como nos trabalhos anteriores.
    const ParametrosAG config1 = {40, 150, 0.80, 0.70, 0.05, 1};
    const ParametrosAG config2 = {80, 300, 1.00, 0.90, 0.10, 2};

    const vector<string> arquivos = {"mochila_4_20", "mochila_100_1000_1"};
    for (const string& nomeArquivo : arquivos) {
        vector<ItemMochila> itens;
        int capacidade = 0;
        if (!carregar_mochila(nomeArquivo, itens, capacidade)) {
            cerr << "Erro: arquivo nao encontrado ou formato invalido: " << nomeArquivo << '\n';
            return 1;
        }

        cout << "\nINSTANCIA: " << nomeArquivo << " | Itens: " << itens.size()
             << " | Capacidade: " << capacidade << '\n';
        executarGeneticoMochila(itens, capacidade, config1);
        executarGeneticoMochila(itens, capacidade, config2);
    }
    return 0;
}
