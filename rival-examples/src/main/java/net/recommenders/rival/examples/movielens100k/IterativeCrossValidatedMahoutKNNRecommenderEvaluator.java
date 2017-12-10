/*
 * Copyright 2015 recommenders.net.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package net.recommenders.rival.examples.movielens100k;

import net.recommenders.rival.core.DataModelIF;
import net.recommenders.rival.core.DataModelUtils;
import net.recommenders.rival.core.Parser;
import net.recommenders.rival.core.SimpleParser;
import net.recommenders.rival.evaluation.metric.ranking.NDCG;
import net.recommenders.rival.evaluation.metric.ranking.Precision;
import net.recommenders.rival.evaluation.strategy.EvaluationStrategy;
import net.recommenders.rival.examples.DataDownloader;
import net.recommenders.rival.recommend.frameworks.RecommenderIO;
import net.recommenders.rival.recommend.frameworks.mahout.GenericRecommenderBuilder;
import net.recommenders.rival.recommend.frameworks.exceptions.RecommenderException;
import net.recommenders.rival.split.parser.MovielensParser;
import net.recommenders.rival.split.splitter.IterativeCrossValidationSplitter;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;
import java.util.List;
import net.recommenders.rival.core.DataModelFactory;
import net.recommenders.rival.evaluation.metric.error.RMSE;

/**
 * RiVal Movielens100k Mahout Example, using 5-fold iterative cross-validation.
 *
 * Note: Adapted from CrossValidatedMahoutKNNRecommenderEvaluator the main
 * difference is the usage of IterativeCrossValidationSplitter instead of
 * CrossValidationSplitter
 *
 * Using IterativeCrossValidationSplitter reduces the memory required to
 * generate data folds.
 *
 * @author <a href="https://github.com/afcarvalho1991">André Carvalho</a>
 */
public final class IterativeCrossValidatedMahoutKNNRecommenderEvaluator {

	/**
	 * Número default posições.
	 */
	public static final int N_FOLDS = 5;
	/**
	 * Número padrão de vizinhos.
	 */
	public static final int NEIGH_SIZE = 50;
	/**
	 * Limite para as métricas de avaliação
	 */
	public static final int AT = 10;
	/**
	 * Limite padrão da relevância.
	 */
	public static final double REL_TH = 3.0;
	/**
	 * Limite padrão de sementes.
	 */
	public static final long SEED = 100L;

	/**
	 * Construtor da classe
	 */
	private IterativeCrossValidatedMahoutKNNRecommenderEvaluator() {
	}

	/**
	 * Método principal responsável inicializar as funcionalidades
	 *
	 * @param args
	 *            the arguments (not used)
	 */
	public static void main(final String[] args) {
		String url = "http://files.grouplens.org/datasets/movielens/ml-100k.zip";
//		String url = "http://github.com/rickanon/TrabalhoUfba/blob/master/ml-100k.zip";
		String folder = "data/ml-100k";
		String modelPath = "data/ml-100k/model/";
		String recPath = "data/ml-100k/recommendations/";
		String dataFile = "data/ml-100k/ml-100k/u.data";
		int nFolds = N_FOLDS;
		prepareSplits(url, nFolds, dataFile, folder, modelPath);
    recommend(nFolds, modelPath, recPath);
		// the strategy files are (currently) being ignored
		prepareStrategy(nFolds, modelPath, recPath, modelPath);
	evaluate(nFolds, modelPath, recPath);
	}

	/** 
	 Faz download 
	 */
	public static void prepareSplits(final String url, final int nFolds, final String inFile, final String folder,
			final String outPath) {
//		DataDownloader dd = new DataDownloader(url, folder);
//		dd.downloadAndUnzip();
//		dd.download();

		boolean perUser = true;
		long seed = SEED;
		Parser<Long, Long> parser = new MovielensParser();

		DataModelIF<Long, Long> data = null;
		try {
			data = parser.parseData(new File(inFile));
		} catch (IOException e) {
			e.printStackTrace();
		}

		new IterativeCrossValidationSplitter<Long, Long>(nFolds, perUser, seed, outPath).split(data);
		File dir = new File(outPath);
		if (!dir.exists()) {
			if (!dir.mkdir()) {
				System.err.println("Directory " + dir + " could not be created");
				return;
			}
		}

	}
//
//	/**
//	 */
	public static void recommend(final int nFolds, final String inPath, final String outPath) {
		for (int i = 0; i < nFolds; i++) {
			org.apache.mahout.cf.taste.model.DataModel trainModel;
			org.apache.mahout.cf.taste.model.DataModel testModel;
			try {
				trainModel = new FileDataModel(new File(inPath + "train_" + i + ".csv"));
				testModel = new FileDataModel(new File(inPath + "test_" + i + ".csv"));
			} catch (IOException e) {
				e.printStackTrace();
				return;
			}

			GenericRecommenderBuilder grb = new GenericRecommenderBuilder();
			String recommenderClass = "org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender";
			String similarityClass = "org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity";
			int neighborhoodSize = NEIGH_SIZE;
			Recommender recommender = null;
			try {
				recommender = grb.buildRecommender(trainModel, recommenderClass, similarityClass, neighborhoodSize);
			} catch (RecommenderException e) {
				e.printStackTrace();
			}

			String fileName = "recs_" + i + ".csv";

			LongPrimitiveIterator users;
			try {
				users = testModel.getUserIDs();
				boolean createFile = true;
				while (users.hasNext()) {
					long u = users.nextLong();
					assert recommender != null;
					List<RecommendedItem> items = recommender.recommend(u, trainModel.getNumItems());
					//
					List<RecommenderIO.Preference<Long, Long>> prefs = new ArrayList<>();
					for (RecommendedItem ri : items) {
						prefs.add(new RecommenderIO.Preference<>(u, ri.getItemID(), ri.getValue()));
					}
					//
					RecommenderIO.writeData(u, prefs, outPath, fileName, !createFile, null);
					createFile = false;
				}
			} catch (TasteException e) {
				e.printStackTrace();
			}
		}
	}

	/**

	 */
	@SuppressWarnings("unchecked")
	public static void prepareStrategy(final int nFolds, final String splitPath, final String recPath,
			final String outPath) {
		for (int i = 0; i < nFolds; i++) {
			File trainingFile = new File(splitPath + "train_" + i + ".csv");
			File testFile = new File(splitPath + "test_" + i + ".csv");
			File recFile = new File(recPath + "recs_" + i + ".csv");
			DataModelIF<Long, Long> trainingModel;
			DataModelIF<Long, Long> testModel;
			DataModelIF<Long, Long> recModel;
			try {
				trainingModel = new SimpleParser().parseData(trainingFile);
				testModel = new SimpleParser().parseData(testFile);
				recModel = new SimpleParser().parseData(recFile);
			} catch (IOException e) {
				e.printStackTrace();
				return;
			}

			Double threshold = REL_TH;
			String strategyClassName = "net.recommenders.rival.evaluation.strategy.UserTest";
			
		
			EvaluationStrategy<Long, Long> strategy = null;
			try {
				strategy = (EvaluationStrategy<Long, Long>) (Class.forName(strategyClassName))
						.getConstructor(DataModelIF.class, DataModelIF.class, double.class)
						.newInstance(trainingModel, testModel, threshold);
			} catch (InstantiationException | IllegalAccessException | NoSuchMethodException | ClassNotFoundException
					| InvocationTargetException e) {
				e.printStackTrace();
			}

			DataModelIF<Long, Long> modelToEval = DataModelFactory.getDefaultModel();
			for (Long user : recModel.getUsers()) {
				assert strategy != null;
				for (Long item : strategy.getCandidateItemsToRank(user)) {
					if (!Double.isNaN(recModel.getUserItemPreference(user, item))) {
						modelToEval.addPreference(user, item, recModel.getUserItemPreference(user, item));
					}
				}
			}
			try {
				DataModelUtils.saveDataModel(modelToEval, outPath + "strategymodel_" + i + ".csv", true, "\t");
			} catch (FileNotFoundException | UnsupportedEncodingException e) {
				e.printStackTrace();
			}
		}
	}

	/**
	
	
	 */
	public static void evaluate(final int nFolds, final String splitPath, final String recPath) {
		double ndcgRes = 0.0;
		double precisionRes = 0.0;
		double rmseRes = 0.0;
		for (int i = 0; i < nFolds; i++) {
			File testFile = new File(splitPath + "test_" + i + ".csv");
			File recFile = new File(recPath + "recs_" + i + ".csv");
			DataModelIF<Long, Long> testModel = null;
			DataModelIF<Long, Long> recModel = null;
			try {
				testModel = new SimpleParser().parseData(testFile);
				recModel = new SimpleParser().parseData(recFile);
			} catch (IOException e) {
				e.printStackTrace();
			}
			NDCG<Long, Long> ndcg = new NDCG<>(recModel, testModel, new int[] { AT });
			ndcg.compute();
			ndcgRes += ndcg.getValueAt(AT);
		

			RMSE<Long, Long> rmse = new RMSE<>(recModel, testModel);
			rmse.compute();
			rmseRes += rmse.getValue();

			Precision<Long, Long> precision = new Precision<>(recModel, testModel, REL_TH, new int[] { AT });
			precision.compute();
			precisionRes += precision.getValueAt(AT);
		}
		System.out.println("NDCG@" + AT + ": " + ndcgRes / nFolds);
		System.out.println("RMSE: " + rmseRes / nFolds);
		System.out.println("P@" + AT + ": " + precisionRes / nFolds);

	}
}


//http://files.grouplens.org/datasets/movielens/ml-100k-README.txt
//
//Main : 
//
//Baixa do repositorio o dataset
//Organiza em pastas os files e models que serão utilizados e começa a inicializar o projeto 
//
//
//
//PrepareSplits
//
//Baixa o arquivo do site do movielens e da unzip depois disso set o arquivo principal e converte para o tipo Movielens
//
//MovielensParser : 
//
//Instancia um novo objeto do tipo Movielens  - Falar sobre o que é o movielens e um resumo do projeto
//
//
//IterativeCrossValidationSplitter
//
//recommender:
//
//recommender = grb.buildRecommender(trainModel, recommenderClass, similarityClass, neighborhoodSize);
//
//Cria uma base de teste e treino. Após isso ele chama o recomender que pega o resultado da base de treino  e chama o  build recomender, com base nisso ele cria um arquivo de recomendação pra cada base de treino.	
//
//
//prepareStrategy
//Instancia os arquivos de treinamento, teste e recomendação como tipo de model pra pode fazer a analise de estrategia para avaliar se os itens são passiveis de tal rank e podem ser utilizados . 
//
//
//Evaluate :
//
//Avalia os resusltados de teste e recomendação e calcula as métricas de acordo com os limites estabelicido e por fim divide o resultado da métrica na posição X pelo numero de folds e obtem se ua " 
