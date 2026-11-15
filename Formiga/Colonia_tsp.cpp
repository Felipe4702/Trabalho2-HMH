#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <random>
#include <chrono>
#include <iomanip>
#include <numeric>
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

struct ResultadoExecucaoTSP {
    int execucao;
    int custo;
    double tempo_ms;
};

// Roleta para escolher uma cidade proporcionalmente aos pesos calculados.
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

// Cada formiga cria uma rota completa a partir de uma cidade inicial aleatoria.
vector<int> construir_rota(const vector<vector<int>>& matriz,
                          const vector<vector<double>>& feromonio,
                          double alpha, double beta, mt19937& gen) {
    int n = (int)matriz.size();
    vector<int> rota;
    rota.reserve(n);
    vector<bool> visitado(n, false);

    uniform_int_distribution<int> sorteio_inicial(0, n - 1);
    int atual = sorteio_inicial(gen);
    rota.push_back(atual);
    visitado[atual] = true;

    while ((int)rota.size() < n) {
        vector<int> candidatos;
        vector<double> pesos;
        for (int j = 0; j < n; j++) {
            if (!visitado[j]) {
                // p_ij proporcional a tau_ij^alpha * (1 / d_ij)^beta.
                // O maximo com 1 evita divisao por zero apos arredondamento.
                double eta = 1.0 / max(1, matriz[atual][j]);
                double peso = pow(feromonio[atual][j], alpha) * pow(eta, beta);
                candidatos.push_back(j);
                pesos.push_back(peso);
            }
        }
        int proxima = selecionar_roleta(candidatos, pesos, gen);
        visitado[proxima] = true;
        rota.push_back(proxima);
        atual = proxima;
    }
    return rota;
}

// Atualizacao offline: evapora todas as trilhas e reforca as rotas das formigas.
void atualizar_feromonio(vector<vector<double>>& feromonio,
                         const vector<vector<int>>& rotas,
                         const vector<int>& custos, const ParametrosACO& p) {
    int n = (int)feromonio.size();
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            feromonio[i][j] *= (1.0 - p.rho);
            feromonio[i][j] = max(1e-12, feromonio[i][j]);
            feromonio[j][i] = feromonio[i][j];
        }
    }

    for (size_t f = 0; f < rotas.size(); f++) {
        const vector<int>& rota = rotas[f];
        if (rota.size() < 2) continue;
        double delta = p.Q / max(1, custos[f]);
        for (size_t i = 0; i < rota.size(); i++) {
            int a = rota[i];
            int b = rota[(i + 1) % rota.size()];
            feromonio[a][b] += delta;
            feromonio[b][a] += delta;
        }
    }
}

// Construcao das rotas, avaliacao e atualizacao de feromonios por iteracao.
vector<int> colonia_de_formigas(const vector<vector<int>>& matriz,
                                const ParametrosACO& p, mt19937& gen) {
    int n = (int)matriz.size();
    vector<vector<double>> feromonio(n, vector<double>(n, p.tau_inicial));
    vector<int> melhor_rota;
    int melhor_custo = numeric_limits<int>::max();

    for (int iter = 0; iter < p.iteracoes; iter++) {
        vector<vector<int>> rotas;
        vector<int> custos;
        rotas.reserve(p.numero_formigas);
        custos.reserve(p.numero_formigas);

        for (int f = 0; f < p.numero_formigas; f++) {
            vector<int> rota = construir_rota(matriz, feromonio, p.alpha, p.beta, gen);
            int custo = avaliar_tsp(rota, matriz);
            if (custo < melhor_custo) {
                melhor_custo = custo;
                melhor_rota = rota;
            }
            rotas.push_back(move(rota));
            custos.push_back(custo);
        }
        atualizar_feromonio(feromonio, rotas, custos, p);
    }
    return melhor_rota;
}

void executarColoniaTSP(const vector<Cidade>& cidades, const vector<vector<int>>& matriz,
                       const ParametrosACO& p) {
    vector<ResultadoExecucaoTSP> resultados(10);
    random_device rd;
    mt19937 gen(rd());
    vector<int> melhor_rota_geral;
    int melhor_custo_geral = numeric_limits<int>::max();

    auto inicio_geral = chrono::steady_clock::now();
    for (int exec = 0; exec < 10; exec++) {
        auto inicio_exec = chrono::steady_clock::now();
        vector<int> melhor_rota = colonia_de_formigas(matriz, p, gen);
        int custo = avaliar_tsp(melhor_rota, matriz);
        auto fim_exec = chrono::steady_clock::now();
        chrono::duration<double, milli> duracao = fim_exec - inicio_exec;

        resultados[exec] = {exec + 1, custo, duracao.count()};
        if (custo < melhor_custo_geral) {
            melhor_custo_geral = custo;
            melhor_rota_geral = melhor_rota;
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
    cout << " TABELA DE RESULTADOS (TSP - COLONIA DE FORMIGAS) - CONFIGURACAO "
         << p.config_id << "\n";
    cout << " (Formigas = " << p.numero_formigas << ", Iteracoes = " << p.iteracoes
         << ", Alpha = " << p.alpha << ", Beta = " << p.beta
         << ", Rho = " << p.rho << ", Q = " << p.Q
         << ", Tau0 = " << p.tau_inicial << ")\n";
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
    const vector<string> arquivos = {"tsp_5", "tsp_51"};
    for (const string& nomeArquivo : arquivos) {
        vector<Cidade> cidades;
        vector<vector<int>> matriz;
        if (!carregar_tsp(nomeArquivo, cidades, matriz)) {
            cerr << "Erro: arquivo nao encontrado ou formato invalido: " << nomeArquivo << '\n';
            return 1;
        }

        int n = (int)cidades.size();
        // Configuracao 1: uma formiga por cidade, como pede o exercicio.
        const ParametrosACO config1 = {120, n, 1.0, 1.0, 0.01, 10.0, 0.1, 1};
        // Configuracao 2: menos formigas e parametros alterados para comparar.
        const ParametrosACO config2 = {200, max(1, n / 2), 1.0, 2.0, 0.05, 10.0, 0.1, 2};

        cout << "\nINSTANCIA: " << nomeArquivo << " | Cidades: " << n << '\n';
        executarColoniaTSP(cidades, matriz, config1);
        executarColoniaTSP(cidades, matriz, config2);
    }
    return 0;
}
