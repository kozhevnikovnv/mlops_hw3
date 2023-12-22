import unittest
from unittest.mock import patch, MagicMock
import app 

class TestApp(unittest.TestCase):
    @patch('app.Repo')
    def test_push_file_to_s3(self, mock_repo):
        mock_repo.return_value = MagicMock()
        response = app.push_file_to_s3('test.txt')
        mock_repo.return_value.add.assert_called_once_with('test.txt')
        mock_repo.return_value.push.assert_called_once_with(['test.txt'])
        self.assertEqual(response, {"status": "Success", "message": "Файл test.txt успешно отправлен в S3"})

if __name__ == '__main__':
    unittest.main()