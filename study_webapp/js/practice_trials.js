let totalImages = 0;
let currentImage = 1;
let table_nums = [40, 60]


// Load the SVG file names from the text file
fetch("js/example_paths.txt")
    .then(response => response.text())
    .then(data => {
        // Split the text file contents into an array of image names
        imageNames = data.trim().split("\n");
        shuffleArray(imageNames)
        totalImages = imageNames.length;
        random_indices = Math.floor(Math.random() * 6)

        // Generate the HTML code for each image container
        let imageContainers = "";
        let j = 0
        let table_i = 0
        for (let i = 0; i < totalImages; i++) {
            imageContainers += `<div id="image-container-${j + 1}" class="image-container">
                              <object data="${imageNames[i]}" type="image/svg+xml"></object>
                           </div>`;
            j += 1;

            // // randomly insert tables into the trials
            // if (random_indices == i) {
            //     imageContainers += `
            //     <div id="image-container-${j + 1}" class="image-container">
            //         <table style="width:12cm;height:6cm;">
            //             <tr>
            //                 <th scope="col" style="font-weight: bold;">${table_nums[0]}</th>
            //                 <th scope="col" style="font-weight: bold;">${table_nums[1]}</th>
            //             </tr>

            //         </table>
            //     </div>
            //     `;
            //     table_i += 1
            //     j += 1
            // }
        }


        // Add the image containers to the DOM
        document.getElementById("image-container").innerHTML = imageContainers;
        document.getElementById(`image-container-${currentImage}`).style.display = "block";

    });

function showNextImage() {

    // Hide the current image container
    document.getElementById(`image-container-${currentImage}`).style.display = "none";

    // Increment the current image counter
    currentImage++;

    console.log(currentImage + "/" + totalImages)

    // If we have reached the end, redirect to the completion page
    if (currentImage == totalImages) {
        document.getElementById('trial-button').setAttribute("onclick", "window.location.href = 'index.html';")
    }

    // Show the next image container
    document.getElementById(`image-container-${currentImage}`).style.display = "block";

    // Update the progress bar
    const progress = (currentImage / totalImages) * 100;
    document.querySelector('.progress').style.width = `${progress}%`;

    if (currentImage > 1) {
        document.getElementById(`p-button`).disabled = false;
    }
}

function showLastImage() {
    // Hide the current image container
    document.getElementById(`image-container-${currentImage}`).style.display = "none";

    // Increment the current image counter
    currentImage--;

    console.log(currentImage + "/" + totalImages)

    // If we have reached the end, redirect to the completion page
    if (currentImage == totalImages) {
        document.getElementById('trial-button').setAttribute("onclick", "window.location.href = 'index.html';")
    }

    // Show the next image container
    document.getElementById(`image-container-${currentImage}`).style.display = "block";

    if (currentImage != totalImages) {
        document.getElementById('trial-button').setAttribute("onclick", "showNextImage()")
    }

    // Update the progress bar
    const progress = (currentImage / totalImages) * 100;
    document.querySelector('.progress').style.width = `${progress}%`;

    if (currentImage == 1) {
        document.getElementById(`p-button`).disabled = true;
    }
}


function shuffleArray(array) {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
    }
    return array;
}



function generate_pairs(ratios) {
    let pairs = []

    for (let i = 0; i < ratios.length; i++) {
        let larger_num = Math.random() * 100
        let smaller_num = larger_num * ratios[i]
        larger_num = larger_num.toFixed(1)
        smaller_num = smaller_num.toFixed(1)

        let coin_flip = Math.floor(Math.random() * 1)
        if (coin_flip == 0) {
            let tmp = larger_num
            larger_num = smaller_num
            smaller_num = tmp
        }
        pairs.push([larger_num, smaller_num])
    }

     return pairs
}
