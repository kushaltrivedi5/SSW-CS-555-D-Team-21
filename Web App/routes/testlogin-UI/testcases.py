import unittest
from selenium import webdriver
 
class TestLoginForm(unittest.TestCase):
 
    def setUp(self):
        iself.driver = webdriver.Chrome()  
        self.driver.get("http://localhost:4200/loginlink")  
 
    def test_valid_email_input(self):
        email_input = self.driver.find_element_by_css_selector("input[type='email']")
        email_input.send_keys("validemail@example.com")
        invalid_email_text = self.driver.find_elements_by_xpath("//div[contains(text(), 'Email is Invalid')]")
        self.assertEqual(len(invalid_email_text), 0)
 
    def test_invalid_email_input(self):
        email_input = self.driver.find_element_by_css_selector("input[type='email']")
        email_input.send_keys("invalidemail")
        invalid_email_text = self.driver.find_elements_by_xpath("//div[contains(text(), 'Email is Invalid')]")
        self.assertNotEqual(len(invalid_email_text), 0)
 
    def test_password_required(self):
        login_button = self.driver.find_element_by_css_selector("button[type='submit']")
        login_button.click()
        password_required_text = self.driver.find_elements_by_xpath("//div[contains(text(), 'Password is required')]")
        self.assertNotEqual(len(password_required_text), 0)
 
    def test_login_button_presence(self):
        login_button = self.driver.find_element_by_css_selector("button[type='submit']")
        self.assertTrue(login_button.is_displayed())
 
    def test_header_presence(self):
        header_text = self.driver.find_element_by_xpath("//span[text()='LogIn']")
        self.assertTrue(header_text.is_displayed())
 
    def test_navigation_sidebar_presence(self):
        navigation_sidebar = self.driver.find_element_by_css_selector(".sidebar")
        self.assertTrue(navigation_sidebar.is_displayed())
 
    def tearDown(self):
        self.driver.quit()
 
if __name__ == "__main__":
    unittest.main()
