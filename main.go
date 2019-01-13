package main

import (
	"encoding/csv"
	"errors"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
	"time"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

// redeNeural contem todas as informações da
// nossa rede neural
type redeNeural struct {
	config  redeNeuralConfig
	pesosOculta *mat.Dense
	biasOculta *mat.Dense
	pesosSaida    *mat.Dense
	biasSaida    *mat.Dense
}

// redeNeuralConfig define como nosssa rede será
type redeNeuralConfig struct {
	neuroniosEntrada  int
	neuroniosSaida int
	neuroniosOculta int
	epocas     int
	taxaAprendizado  float64
}

// novaRedeNeural retorna a instancia de uma rede neural
func novaRedeNeural(config redeNeuralConfig) *redeNeural {
	return &redeNeural{config: config}
}

// sumaNoEixo soma uma matriz ao longo
// de um eixo em especifico (0 ou 1)
// preservando a outra dimenção
func sumaNoEixo(eixo int, m *mat.Dense) (*mat.Dense, error) {
	
	numLinhas, numColunas := m.Dims()
	
	var saida *mat.Dense
	
	switch eixo {
	case 0:
		data := make([]float64, numColunas)
		for i := 0; i < numColunas; i++ {
			col := mat.Col(nil, i, m)
			data[i] = floats.Sum(col)
		}
		saida = mat.NewDense(1, numColunas, data)
	case 1:
		data := make([]float64, numLinhas)
		for i := 0; i < numLinhas; i++ {
			row := mat.Row(nil, i, m)
			data[i] = floats.Sum(row)
		}
		saida = mat.NewDense(numLinhas, 1, data)
	default:
		return nil, errors.New("Eixo ivalido, deve ser 0 ou 1")
	}
	
	return saida, nil
}

// FUnção de ativação
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// derivada da sigmoid
func sigmoidDerivada(x float64) float64 {
	return x * (1.0 - x)
}

// esse metodo executa todo o feed forward e o back propagate
func (rn *redeNeural) backpropagate(x, y, pesosOculta, biasOculta, pesosSaida, biasSaida, saida *mat.Dense) error {

	//treinamento
	for i := 0; i < rn.config.epocas; i++ {
		// Complete the feed forward process.
		EntradasLayerOculta := new(mat.Dense)
		EntradasLayerOculta.Mul(x, pesosOculta)
		addBiasOculta := func(_, col int, v float64) float64 { return v + biasOculta.At(0, col) }
		EntradasLayerOculta.Apply(addBiasOculta, EntradasLayerOculta)

		AtivacaoLayerOculta := new(mat.Dense)
		aplicaSigmoid := func(_, _ int, v float64) float64 { return sigmoid(v) }
		AtivacaoLayerOculta.Apply(aplicaSigmoid, EntradasLayerOculta)

		saidaCamadaEntrada := new(mat.Dense)
		saidaCamadaEntrada.Mul(AtivacaoLayerOculta, pesosSaida)
		addBiasSaida := func(_, col int, v float64) float64 { return v + biasSaida.At(0, col) }
		saidaCamadaEntrada.Apply(addBiasSaida, saidaCamadaEntrada)
		saida.Apply(aplicaSigmoid, saidaCamadaEntrada)

		// Complete the backpropagation.
		ErroRede := new(mat.Dense)
		ErroRede.Sub(y, saida)

		derivadaCamadaSaida := new(mat.Dense)
		aplicasigmoidDerivada := func(_, _ int, v float64) float64 { return sigmoidDerivada(v) }
		derivadaCamadaSaida.Apply(aplicasigmoidDerivada, saida)
		slopeHiddenLayer := new(mat.Dense)
		slopeHiddenLayer.Apply(aplicasigmoidDerivada, AtivacaoLayerOculta)

		deltaSaida := new(mat.Dense)
		deltaSaida.MulElem(ErroRede, derivadaCamadaSaida)
		errorCamadaOculta := new(mat.Dense)
		errorCamadaOculta.Mul(deltaSaida, pesosSaida.T())

		deltaCamadaOculta := new(mat.Dense)
		deltaCamadaOculta.MulElem(errorCamadaOculta, slopeHiddenLayer)

		// Aajuste dos pesos e bias
		pesosSaidaAjustato := new(mat.Dense)
		pesosSaidaAjustato.Mul(AtivacaoLayerOculta.T(), deltaSaida)
		pesosSaidaAjustato.Scale(rn.config.taxaAprendizado, pesosSaidaAjustato)
		pesosSaida.Add(pesosSaida, pesosSaidaAjustato)

		biasSaidaAjustado, err := sumaNoEixo(0, deltaSaida)
		if err != nil {
			return err
		}
		biasSaidaAjustado.Scale(rn.config.taxaAprendizado, biasSaidaAjustado)
		biasSaida.Add(biasSaida, biasSaidaAjustado)

		pesosOcultaAjustado := new(mat.Dense)
		pesosOcultaAjustado.Mul(x.T(), deltaCamadaOculta)
		pesosOcultaAjustado.Scale(rn.config.taxaAprendizado, pesosOcultaAjustado)
		pesosOculta.Add(pesosOculta, pesosOcultaAjustado)

		biasOcultaAjustado, err := sumaNoEixo(0, deltaCamadaOculta)
		if err != nil {
			return err
		}
		biasOcultaAjustado.Scale(rn.config.taxaAprendizado, biasOcultaAjustado)
		biasOculta.Add(biasOculta, biasOcultaAjustado)
	}

	return nil
}

// inicia a rede neural, treina
func (rn *redeNeural) treinar(x, y *mat.Dense) error {
	
	// Initialize biases/weights.
	aleatoriedade := rand.NewSource(time.Now().UnixNano())
	geradorAleatoriedade := rand.New(aleatoriedade)
	
	pesosOculta := mat.NewDense(rn.config.neuroniosEntrada, rn.config.neuroniosOculta, nil)
	biasOculta := mat.NewDense(1, rn.config.neuroniosOculta, nil)
	pesosSaida := mat.NewDense(rn.config.neuroniosOculta, rn.config.neuroniosSaida, nil)
	biasSaida := mat.NewDense(1, rn.config.neuroniosSaida, nil)
	
	pesosOcultaRaw := pesosOculta.RawMatrix().Data
	biasOcultaRaw := biasOculta.RawMatrix().Data
	pesosSaidaRaw := pesosSaida.RawMatrix().Data
	biasSaidaRaw := biasSaida.RawMatrix().Data
	
	for _, param := range [][]float64{
		pesosOcultaRaw,
		biasOcultaRaw,
		pesosSaidaRaw,
		biasSaidaRaw,
	} {
		for i := range param {
			param[i] = geradorAleatoriedade.Float64()
		}
	}
	
	// Define a saida da rede neural
	saida := new(mat.Dense)

	if err := rn.backpropagate(x, y, pesosOculta, biasOculta, pesosSaida, biasSaida, saida); err != nil {
		return err
	}
	
	// Pesos e bias definidos
	rn.pesosOculta = pesosOculta
	rn.biasOculta = biasOculta
	rn.pesosSaida = pesosSaida
	rn.biasSaida = biasSaida


	return nil
}

//  Faz a classificação/predição de um valor cujo o label é desconhecido
func (rn *redeNeural) Classifique(x *mat.Dense) (*mat.Dense, error) {


	if rn.pesosOculta == nil || rn.pesosSaida == nil {
		return nil, errors.New("Os pesos estão vazios")
	}
	if rn.biasOculta == nil || rn.biasSaida == nil {
		return nil, errors.New("Os bias estão vazios")
	}

	saida := new(mat.Dense)

	// processo de feed forwarde com os pesos e bias que já obtemos 
	// no processo de treinamento
	EntradasLayerOculta := new(mat.Dense)
	EntradasLayerOculta.Mul(x, rn.pesosOculta)
	addBiasOculta := func(_, col int, v float64) float64 { return v + rn.biasOculta.At(0, col) }
	EntradasLayerOculta.Apply(addBiasOculta, EntradasLayerOculta)

	AtivacaoLayerOculta := new(mat.Dense)
	aplicaSigmoid := func(_, _ int, v float64) float64 { return sigmoid(v) }
	AtivacaoLayerOculta.Apply(aplicaSigmoid, EntradasLayerOculta)

	saidaCamadaEntrada := new(mat.Dense)
	saidaCamadaEntrada.Mul(AtivacaoLayerOculta, rn.pesosSaida)
	addBiasSaida := func(_, col int, v float64) float64 { return v + rn.biasSaida.At(0, col) }
	saidaCamadaEntrada.Apply(addBiasSaida, saidaCamadaEntrada)
	saida.Apply(aplicaSigmoid, saidaCamadaEntrada)

	return saida, nil
}

func LerDados(fileName string) (*mat.Dense, *mat.Dense) {
	// Abra os arquivo do nosso dataset
	f, err := os.Open(fileName)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	reader := csv.NewReader(f)
	reader.FieldsPerRecord = 7

	// Leia todas as linhas do CSV
	rawCSVData, err := reader.ReadAll()
	if err != nil {
		log.Fatal(err)
	}

	// dadosEntrada e labelsEntrada irão armazenar
	// todos os dados do nosso dataset
	dadosEntrada := make([]float64, 4*len(rawCSVData))
	labelsEntrada := make([]float64, 3*len(rawCSVData))

	// vaiaveis de controle
	var entradasIndex int
	var labelsIndex int

	// Mova-se pelas as linhas
	for idx, record := range rawCSVData {

		// Pule a primeira linha do dataset
		if idx == 0 {
			continue
		}

		// Passe pelas colunas
		for i, val := range record {

			// Converta o valor para float
			valorConvertido, err := strconv.ParseFloat(val, 64)
			if err != nil {
				log.Fatal(err)
			}

			// adicione para labelsEntrada se for relevante
			if i == 4 || i == 5 || i == 6 {
				labelsEntrada[labelsIndex] = valorConvertido
				labelsIndex++
				continue
			}

			dadosEntrada[entradasIndex] = valorConvertido
			entradasIndex++
		}
	}
	entradas := mat.NewDense(len(rawCSVData), 4, dadosEntrada)
	labels := mat.NewDense(len(rawCSVData), 3, labelsEntrada)
	return entradas, labels
}

func main() {
	
	// Obter os dadaos para treino
	entradas, labels := LerDados("data/train.csv")
	
	// Definindo com será a arquitetura da nossa redeneural
	config := redeNeuralConfig{
		neuroniosEntrada:  4,
		neuroniosSaida: 3,
		neuroniosOculta: 3,
		epocas:     5000,
		taxaAprendizado:  0.3,
	}
	
	// Treine a rede neural
	network := novaRedeNeural(config)
	if err := network.treinar(entradas, labels); err != nil {
		log.Fatal(err)
	}
	
	// Obter os dados para treino
	testInputs, testLabels := LerDados("data/test.csv")
	
	// faça as prediçoes usando os pesos e bias do nosso treinamento
	predicoes, err := network.Classifique(testInputs)
	if err != nil {
		log.Fatal(err)
	}
	
	// Calcule a acuracia da nossa rede
	var posNeg int
	numPreds, _ := predicoes.Dims()
	for i := 0; i < numPreds; i++ {
	
		// Pegue o label
		linhaLabel := mat.Row(nil, i, testLabels)
		var predicao int
		for idx, label := range linhaLabel {
			if label == 1.0 {
				predicao = idx
				break
			}
		}
	
		// Acumulando os valores acertados
		if predicoes.At(i, predicao) == floats.Max(mat.Row(nil, i, predicoes)) {
			posNeg++
		}
	}
	
	// Calculando a acuracia
	accuracy := float64(posNeg) / float64(numPreds)
	fmt.Printf("\nAccuracy = %0.2f\n\n", accuracy)
}
