import pytest
import monica as md


# This fixture initializes the MonicaFunc class for each test
@pytest.fixture
def classFunc():
    return md.MonicaFunc()

class TestMonicaPossTest:
    def test_neg_test(self, classFunc):
        # Use the classFunc fixture to get an instance of MonicaFunc
        result = classFunc.sentimentAn("i feel so bad about myself!!")
        assert result == "The user seems to be sad"

    def test_pos_test(self, classFunc):
        # Use the classFunc fixture to get an instance of MonicaFunc
        result = classFunc.sentimentAn("i feel so happy about myself!!")
        assert result == "The user seems to be happy"

        
