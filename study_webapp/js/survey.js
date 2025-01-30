// set trial count to the session storage
let indices = [0,1,2,3]
const trialCount = {'lines': indices, 'areas': indices}
const countString = JSON.stringify(trialCount)
localStorage.setItem("trialCount", countString)

let form = document.getElementById("survey")
let types = Object.keys(trialCount)

// redirect and randomize the first trial
let firstTrial = types[Math.floor(Math.random() * 2)]
form.setAttribute("action", firstTrial + '.html')

// validation function for the form
function validation() {
    // var id = document.forms.survey_questions.prolific_id.value;
    // var age = document.forms.survey_questions.age.value;
    // var gender = document.getElementById("gender").selectedIndex
    // var education = document.getElementById("education").selectedIndex
    // var race = document.getElementById("race").selectedIndex
    // var visfam = document.getElementById("visfam").selectedIndex
    // var stockfam = document.getElementById("stockfam").selectedIndex

    // if (id == "") {
    // window.alert("Please enter your Prolific ID.");
    // return false;
    // } else if (age == "") {
    // alert("Please enter your age.");
    // return false;
    // } else if (race == 0) {
    // window.alert("Please select your race/ethnicity.");
    // return false;
    // } else if (gender == 0) {
    // window.alert("Please select your gender.");
    // return false;
    // } else if (education == 0) {
    // window.alert("Please select your education.");
    // return false;
    // } else if (visfam == 0) {
    // window.alert("Please select your familiarity with data visualization.");
    // return false;
    // } else if(stockfam == 0) {
    // window.alert("Please select your experience with stock market.");
    // return false;
    // }
    // window.location.replace("www.google.com");

    // return true;

}


function getRandomAndRemove(arr) {
  const randomIndex = Math.floor(Math.random() * arr.length); // Generate a random index within the array
  const removedElement = arr.splice(randomIndex, 1); // Remove the element at the random index
  return removedElement[0]; // Return the removed element
}

function maybeSwitchElements(element1, element2) {
  if (Math.random() < 0.5) {
    const parent1 = element1.parentNode;
    const sibling1 = element1.nextSibling === element2 ? element1 : element1.nextSibling;
    element2.parentNode.insertBefore(element1, element2);
    parent1.insertBefore(element2, sibling1);
  }
}
