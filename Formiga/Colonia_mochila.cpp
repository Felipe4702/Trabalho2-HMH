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

struct SolucaoMochila {
    vector<int> itens_selecionados;
    int valor = 0;
    int peso = 0;
};

struct ParametrosACO {
    int iteracoes;
    int numero_formigas;
    double alpha;
    double beta;
    double rho;
    double Q;
    double tau_inicial;
    int config_id;
};

struct ResultadoExecucaoMochila {
    int execucao;
    int valor_obj;
    int peso_total;
    double tempo_ms;
};

// Roleta para escolher um item proporcionalmente aos pesos calculados.
int selecionar_roleta(const vector<int>& candidatos,
                      const vector<double>& pesos, mt19937& gen) {
    double soma = accumulate(pesos.begin(), pesos.end(), 0.0);
    if (!(soma > 0.0)) {
        uniform_int_distribution<int> sorteio(0, (int)candidatos.size() - 1);
        return candidatos[sorteio(gen)];
    }

    uniform_real_distribution<double> sorteio(0.0, soma);
    double alvo = sorteio(gen);
    double acumulado = 0.0;
    for (size_t i = 0; i < candidatos.size(); i++) {
        acumulado += pesos[i];
        if (alvo <= acumulado) return candidatos[i];
    }
    return candidatos.back();
}

// Cada formiga monta uma mochila, escolhendo apenas itens que ainda cabem.
// A atratividade eta considera a relacao entre valor e peso de cada item.
SolucaoMochila construir_solucao(const vector<ItemMochila>& itens, int cap,
                                 const vector<double>& atratividade,
                                 mt19937& gen) {
    SolucaoMochila solucao;
    solucao.itens_selecionados.assign(itens.size(), 0);

    while (true) {
        vector<int> candidatos;
        vector<double> pesos;
        for (int i = 0; i < (int)itens.size(); i++) {
            if (solucao.itens_selecionados[i] == 0 &&
                itens[i].valor > 0 && solucao.peso + itens[i].peso <= cap) {
                candidatos.push_back(i);
                pesos.push_back(atratividade[i]);
            }
        }
        if (candidatos.empty()) break;

        int escolhido = selecionar_roleta(candidatos, pesos, gen);
        solucao.itens_selecionados[escolhido] = 1;
        solucao.peso += itens[escolhido].peso;
        solucao.valor += itens[escolhido].valor;
    }
    return solucao;
}

// Atualizacao offline: evapora as trilhas e reforca os itens escolhidos.
// Para maximizar, o deposito cresce com o valor da mochila construida.
void atualizar_feromonio(vector<double>& feromonio,
                         const vector<SolucaoMochila>& solucoes,
                         const ParametrosACO& p, int soma_valores) {
    for (double& tau : feromonio) {
        tau *= (1.0 - p.rho);
        tau = max(1e-12, tau);
    }

    for (const SolucaoMochila& solucao : solucoes) {
        double delta = p.Q * (double)solucao.valor /
                       max(1, soma_valores) / p.numero_formigas;
        for (int i = 0; i < (int)feromonio.size(); i++) {
            if (solucao.itens_selecionados[i] == 1) feromonio[i] += delta;
        }
    }
}

// Construcao das mochilas, avaliacao e atualizacao de feromonios por iteracao.
SolucaoMochila colonia_de_formigas(const vector<ItemMochila>& itens, int cap,
                                  const ParametrosACO& p, mt19937& gen) {
    vector<double> feromonio(itens.size(), p.tau_inicial);
    SolucaoMochila melhor_solucao;
    int soma_valores = 0;
    for (const ItemMochila& item : itens) soma_valores += item.valor;

    // A parte heuristica nao muda durante as iteracoes.
    vector<double> heuristica(itens.size());
    for (int i = 0; i < (int)itens.size(); i++) {
        double eta = (double)itens[i].valor / max(1, itens[i].peso);
        heuristica[i] = pow(eta, p.beta);
    }

    for (int iter = 0; iter < p.iteracoes; iter++) {
        vector<double> atratividade(itens.size());
        for (int i = 0; i < (int)itens.size(); i++) {
            // p_i proporcional a tau_i^alpha * (valor_i / peso_i)^beta.
            atratividade[i] = pow(feromonio[i], p.alpha) * heuristica[i];
        }

        vector<SolucaoMochila> solucoes;
        solucoes.reserve(p.numero_formigas);
        for (int f = 0; f < p.numero_formigas; f++) {
            SolucaoMochila solucao = construir_solucao(itens, cap, atratividade, gen);
            if (solucao.valor > melhor_solucao.valor) melhor_solucao = solucao;
            solucoes.push_back(move(solucao));
        }
        atualizar_feromonio(feromonio, solucoes, p, soma_valores);
    }
    return melhor_solucao;
}

void executarColoniaMochila(const vector<ItemMochila>& itens, int cap,
                           const ParametrosACO& p) {
    vector<ResultadoExecucaoMochila> resultados(10);
    random_device rd;
    mt19937 gen(rd());
    SolucaoMochila melhor_solucao_geral;
    melhor_solucao_geral.valor = -1;

    auto inicio_geral = chrono::steady_clock::now();
    for (int exec = 0; exec < 10; exec++) {
        auto inicio_exec = chrono::steady_clock::now();
        SolucaoMochila melhor = colonia_de_formigas(itens, cap, p, gen);
        auto fim_exec = chrono::steady_clock::now();
        chrono::duration<double, milli> duracao = fim_exec - inicio_exec;

        int peso_verificado = 0;
        int valor_verificado = avaliar_mochila(melhor.itens_selecionados, itens,
                                               cap, peso_verificado);
        resultados[exec] = {exec + 1, valor_verificado, peso_verificado, duracao.count()};
        if (valor_verificado > melhor_solucao_geral.valor) {
            melhor_solucao_geral = move(melhor);
        }
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
    cout << " TABELA DE RESULTADOS (MOCHILA - COLONIA DE FORMIGAS) - CONFIGURACAO "
         << p.config_id << "\n";
    cout << " (Formigas = " << p.numero_formigas << ", Iteracoes = " << p.iteracoes
         << ", Alpha = " << p.alpha << ", Beta = " << p.beta
         << ", Rho = " << p.rho << ", Q = " << p.Q
         << ", Tau0 = " << p.tau_inicial << ")\n";
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
        if (melhor_solucao_geral.itens_selecionados[i] == 1) {
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
    const vector<string> arquivos = {"mochila_4_20", "mochila_100_1000_1"};
    for (const string& nomeArquivo : arquivos) {
        vector<ItemMochila> itens;
        int capacidade = 0;
        if (!carregar_mochila(nomeArquivo, itens, capacidade)) {
            cerr << "Erro: arquivo nao encontrado ou formato invalido: " << nomeArquivo << '\n';
            return 1;
        }

        int n = (int)itens.size();
        const ParametrosACO config1 = {120, min(n, 24), 1.0, 1.0, 0.01, 10.0, 0.1, 1};
        const ParametrosACO config2 = {200, max(2, min(n / 2, 12)), 1.0, 2.0, 0.05, 10.0, 0.1, 2};

        cout << "\nINSTANCIA: " << nomeArquivo << " | Itens: " << itens.size()
             << " | Capacidade: " << capacidade << '\n';
        executarColoniaMochila(itens, capacidade, config1);
        executarColoniaMochila(itens, capacidade, config2);
    }
    return 0;
}
