from allennlp.common import Params
import numpy as np
from numpy.testing import assert_allclose
import pytest

from contexteval.common.custom_test_case import CustomTestCase
from contexteval.contextualizers import PrecomputedContextualizer
from contexteval.data.dataset_readers import EventFactualityDatasetReader


class TestEventFactualityDatasetReader():
    data_path = CustomTestCase.FIXTURES_ROOT / "data" / "event_factuality" / "ithappened.json"
    contextualizer_path = (CustomTestCase.FIXTURES_ROOT / "contextualizers" /
                           "precomputed_elmo" / "elmo_layers_all.hdf5")

    @pytest.mark.parametrize('lazy', (True, False))
    @pytest.mark.parametrize('use_contextualizer', (True, False))
    @pytest.mark.parametrize('include_raw_tokens', (True, False))
    def test_read_from_file(self, lazy, use_contextualizer, include_raw_tokens):
        # Set up contextualizer, if using.
        contextualizer = None
        if use_contextualizer:
            contextualizer = PrecomputedContextualizer(self.contextualizer_path)
        reader = EventFactualityDatasetReader(
            include_raw_tokens=include_raw_tokens,
            contextualizer=contextualizer,
            lazy=lazy)
        instances = list(reader.read(str(self.data_path)))
        assert len(instances) == 15

        # First read instance
        instance = instances[0]
        fields = instance.fields
        if include_raw_tokens:
            assert [token.metadata for token in fields["raw_tokens"].field_list] == [
                'Joe', 'asked', 'that', 'you', 'fax', 'the', 'revised', 'gtee',
                'wording', 'that', 'has', 'been', 'agreed', '(', 'I', 'believe', 'it',
                'was', 'our', 'agreeing', 'to', 'reduce', 'the', 'claim', 'period',
                'from', '15', 'days', 'down', 'to', '5', ')', 'and', 'the', 'new', 'l/c',
                'wording', '(', 'drops', 'the', '2', 'day', 'period', 'to', 'replace',
                'an', 'l/c', 'with', 'a', 'different', 'bank', 'if', 'the', 'first',
                'refuses', 'to', 'pay', ')', '.']
        assert_allclose(fields["label_indices"].array,
                        np.array([15, 19, 21, 38, 44, 54, 56, 1, 4, 6, 12]))
        if use_contextualizer:
            assert_allclose(
                fields["token_representations"].array[:, :2],
                np.array([[-0.36685583, -1.0248482], [-0.08719254, -0.12769365],
                          [0.64198303, 0.7540561], [-2.480215, 0.04852793],
                          [0.05662279, 0.19068614], [0.8952136, 0.18563624],
                          [0.61201894, -0.21791479], [0.16892922, -0.79595846],
                          [-0.27208328, -0.13422441], [0.04730925, -0.43866983],
                          [-0.18922694, 0.41402912], [-1.2735212, -0.7098247],
                          [-0.35325307, -0.1886746], [0.24240366, -0.2627995],
                          [-2.657272, -0.85991454], [-0.19721821, -0.28280562],
                          [-1.2974384, -1.5685275], [-0.17114338, -1.3488747],
                          [-0.14475444, -1.3091846], [-0.9352702, -0.42290983],
                          [-1.9790481, -0.19222577], [-0.7576624, -1.3168397],
                          [0.04005039, -0.9087254], [-1.1224419, -1.2120944],
                          [-1.1654481, -1.2385485], [-0.53110546, -0.37541062],
                          [-0.43803376, -0.5062414], [-1.0063732, -1.4231381],
                          [-1.6299391, -0.08710647], [-0.4013245, 1.336797],
                          [-0.31591064, 0.11186421], [-0.9240766, -0.19987631],
                          [-0.91462064, -0.2551515], [0.48850712, -0.05782498],
                          [0.26612586, -0.7230994], [-0.00594145, -1.11585],
                          [-0.82872486, -0.6029454], [0.10594115, 0.6299722],
                          [-0.23010078, 0.5210506], [0.57265085, -0.76853454],
                          [-0.2151854, 0.1495785], [-0.5665817, 0.10349956],
                          [-0.0619593, 0.15140474], [0.47662088, 0.9349986],
                          [0.4795642, 0.4577945], [0.3688566, -0.06091809],
                          [0.29802012, -0.25112373], [0.8288579, 0.28962702],
                          [0.90991616, 0.24866864], [-0.2174969, -1.3967221],
                          [-0.26998952, -1.2395245], [0.40867922, 0.41572857],
                          [0.34937006, -0.21592987], [0.02204479, -1.1068783],
                          [-0.81269974, -0.71383244], [-1.6719012, -0.24751332],
                          [-0.7133447, -0.9015558], [-0.36663392, 0.00226176],
                          [-0.66520894, 0.02220622]]),
                rtol=1e-4)
        assert_allclose(fields["labels"].array,
                        np.array([2.625, 2.625, -1.125, -1.125, -1.5, -2.25, 2.25, 3.0,
                                  -2.625, 2.625, 3.0]))
        assert len(fields["label_indices"].array) == len(fields["labels"].array)

        # Second read instance
        instance = instances[1]
        fields = instance.fields
        if include_raw_tokens:
            assert [token.metadata for token in fields["raw_tokens"].field_list] == [
                'you', 'need', 'to', 'do', 'that', 'for', 'a', 'least', 'a', 'week',
                ',', 'or', 'else', 'this', 'type', 'of', 'territorial', 'fighting',
                'will', 'happen', '.']

        assert_allclose(fields["label_indices"].array, np.array([19, 1, 3]))
        if use_contextualizer:
            assert_allclose(
                fields["token_representations"].array[:, :2],
                np.array([[-1.8767732, 0.017495558], [0.6025951, -0.20260233],
                          [-1.2275248, -0.09547685], [0.77881116, 0.58967566],
                          [-0.48033506, -0.7382006], [0.81346595, -0.5705998],
                          [0.4814891, 0.13833025], [0.29612598, 0.4745674],
                          [1.047016, -0.10455979], [-0.42458856, -1.1668162],
                          [-0.12459692, 1.0916736], [0.28291142, 0.17336448],
                          [-0.08204004, 0.6720216], [-0.55279577, -0.3378092],
                          [0.046703815, 0.0627833], [-0.17136925, 0.07279006],
                          [-0.61967653, -0.36650854], [0.22994132, -0.17982215],
                          [-0.039243788, -0.19590409], [1.0741227, -0.46452063],
                          [-0.99690104, -0.20734516]]),
                rtol=1e-4)
        assert_allclose(fields["labels"].array,
                        np.array([-2.25, -2.25, -2.25]))
        assert len(fields["label_indices"].array) == len(fields["labels"].array)

        # Third read instance
        instance = instances[2]
        fields = instance.fields
        if include_raw_tokens:
            assert [token.metadata for token in fields["raw_tokens"].field_list] == [
                'The', 'police', 'commander', 'of', 'Ninevah', 'Province', 'announced',
                'that', 'bombings', 'had', 'declined', '80', 'percent', 'in', 'Mosul',
                ',', 'whereas', 'there', 'had', 'been', 'a', 'big', 'jump', 'in', 'the',
                'number', 'of', 'kidnappings', '.']

        assert_allclose(fields["label_indices"].array, np.array([6, 10]))
        if use_contextualizer:
            assert_allclose(
                fields["token_representations"].array[:, :2],
                np.array([[0.30556452, -0.03631544], [-0.408598, -0.06455576],
                          [-1.5492078, -0.54190993], [-0.2587847, -0.2953575],
                          [-0.73995805, 0.4683098], [-0.7363724, 0.79993117],
                          [-0.16775642, 1.7011331], [0.41263822, 1.7853746],
                          [0.20824504, 0.94526154], [0.38856286, 0.55201274],
                          [0.21549016, 0.29676253], [0.19024657, 1.6858654],
                          [-0.3601446, 0.9940252], [0.06638061, 2.2731574],
                          [-0.83813465, 1.996573], [-0.2564547, 1.3016648],
                          [1.2408254, 1.2657689], [1.2441401, 0.26394492],
                          [1.2946486, 0.4354594], [1.5132289, -0.28065175],
                          [1.3383818, 0.99084306], [1.04397, -0.52631915],
                          [1.026963, 0.8950106], [1.1683758, 0.3674168],
                          [1.568187, 0.60855913], [0.00261295, 1.0362052],
                          [1.0013494, 1.1375219], [0.46779868, 0.85086995],
                          [-0.23202378, -0.3398294]]),
                rtol=1e-4)
        assert_allclose(fields["labels"].array, np.array([3.0, 3.0]))

    @pytest.mark.parametrize('lazy', (True, False))
    @pytest.mark.parametrize('include_raw_tokens', (True, False))
    @pytest.mark.parametrize('max_instances', (1, 2, 0.5, 0.75, 1.0))
    def test_reproducible_with_and_without_contextualization(self, lazy, include_raw_tokens,
                                                             max_instances):
        uncontextualized_params = Params({
            "lazy": lazy,
            "max_instances": max_instances,
            "include_raw_tokens": include_raw_tokens})
        uncontextualized_reader = EventFactualityDatasetReader.from_params(
            uncontextualized_params)
        uncontextualized_instances = list(uncontextualized_reader.read(str(self.data_path)))

        contextualized_params = Params({
            "lazy": lazy,
            "max_instances": max_instances,
            "include_raw_tokens": include_raw_tokens,
            "contextualizer": {
                "type": "precomputed_contextualizer",
                "representations_path": self.contextualizer_path
            }})
        contextualized_reader = EventFactualityDatasetReader.from_params(
            contextualized_params)
        contextualized_instances = list(contextualized_reader.read(str(self.data_path)))
        # Assert they are the same
        for uncontextualized_instance, contextualized_instance in zip(uncontextualized_instances,
                                                                      contextualized_instances):
            if include_raw_tokens:
                assert ([token.metadata for token in uncontextualized_instance.fields["raw_tokens"].field_list] ==
                        [token.metadata for token in contextualized_instance.fields["raw_tokens"].field_list])
            assert_allclose(uncontextualized_instance.fields["label_indices"].array,
                            contextualized_instance.fields["label_indices"].array)
            assert_allclose(uncontextualized_instance.fields["labels"].array,
                            contextualized_instance.fields["labels"].array)
            contextualized_extra_keys = list(set(contextualized_instance.fields.keys()) -
                                             set(uncontextualized_instance.fields.keys()))
            assert (set(contextualized_extra_keys) == set(["token_representations"]))

    @pytest.mark.parametrize('lazy', (True, False))
    @pytest.mark.parametrize('use_contextualizer', (True, False))
    @pytest.mark.parametrize('include_raw_tokens', (True, False))
    @pytest.mark.parametrize('max_instances', (1, 2, 0.5, 0.75, 1.0))
    def test_truncation(self, lazy, use_contextualizer, include_raw_tokens,
                        max_instances):
        # Set up contextualizer, if using.
        contextualizer = None
        if use_contextualizer:
            contextualizer = PrecomputedContextualizer(self.contextualizer_path)
        reader = EventFactualityDatasetReader(
            include_raw_tokens=include_raw_tokens,
            contextualizer=contextualizer,
            max_instances=max_instances,
            lazy=lazy)
        instances = list(reader.read(str(self.data_path)))
        num_total_instances = 15
        max_instances_to_num_instances = {
            int(1): 1,
            int(2): 2,
            0.5: int(num_total_instances * 0.5),
            0.75: int(num_total_instances * 0.75),
            1.0: num_total_instances}
        assert len(instances) == max_instances_to_num_instances[max_instances]
