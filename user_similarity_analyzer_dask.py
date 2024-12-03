# user_similarity_analyzer_dask.py
from file_writer import FileWriter
from similarity_calculator import SimilarityCalculator
import dask
import dask.bag as db
import dask.array as da

class UserSimilarityAnalyzerdask:
    @staticmethod
    def generate_key_value_pairs(data):
        key_value_pairs = []

        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    key_value_pairs.extend(UserSimilarityAnalyzerdask.generate_key_value_pairs(item))
        elif isinstance(data, dict):
            for module, roles_data in data.items():
                if isinstance(roles_data, list):
                    for role_data in roles_data:
                        if isinstance(role_data, dict):
                            temp_item = {k: v for k, v in role_data.items() if k != 'id'}
                            key_value_pairs.append((module, None, None, None, temp_item))
                elif isinstance(roles_data, dict):
                    for role, role_data in roles_data.items():
                        if isinstance(role_data, list):
                            role_index = len(key_value_pairs) + 1
                            for user_index, item in enumerate(role_data, start=1):
                                temp_item = {k: v for k, v in item.items() if k != 'id'}
                                key_value_pairs.append((module, role, role_index, user_index, temp_item))

        return key_value_pairs

    @staticmethod
    def calculate_similarity_score(pair, embeddings_cache, similarity_calculator):
        module1, role1, role1_index, user1_index, user1 = pair[0]
        module2, role2, role2_index, user2_index, user2 = pair[1]

        similarities = []
        for key1, value1 in user1.items():
            for key2, value2 in user2.items():
                embedding1 = similarity_calculator.get_word_embedding(value1)
                embedding2 = similarity_calculator.get_word_embedding(value2)
                similarity_score = similarity_calculator.calculate_cosine_similarity(embedding1, embedding2)
                similarities.append((i, j, module1, role1, role1_index, user1_index, key1, value1,
                                     module2, role2, role2_index, user2_index, key2, value2, similarity_score))

        return similarities

    @staticmethod
    def calculate_similarity_scores(all_key_value_pairs, embeddings_cache, nlp_model, output_filename, threshold):
        file_writer = FileWriter()  # Create an instance of FileWriter
        similarity_calculator = SimilarityCalculator()  # Create an instance of SimilarityCalculator

        # Convert all_key_value_pairs to a Dask Bag for parallel processing
        pairs = [(pair1, pair2) for i, pair1 in enumerate(all_key_value_pairs) for j, pair2 in enumerate(all_key_value_pairs) if pair1 != pair2 and pair1[1] != pair2[1]]
        dask_bag = db.from_sequence(pairs)

        # Process each pair using Dask
        results = dask_bag.map(lambda pair: UserSimilarityAnalyzerdask.calculate_similarity_score(pair, embeddings_cache, similarity_calculator)).compute()

        # Write results to file
        with open(output_filename, "w") as file:
            for result in results:
                for i, j, module1, role1, role1_index, user1_index, key1, value1, module2, role2, role2_index, user2_index, key2, value2, similarity_score in result:
                    if similarity_score >= threshold:
                        file_writer.write_similarity_scores(output_filename, i, module1, role1, role1_index, user1_index,
                                                            key1, value1, j, module2, role2, role2_index, user2_index,
                                                            key2, value2, similarity_score, threshold)

