let totalImages = 0;
let currentImage = 1;
let table_nums = [[24.3, 30.3],[12.7, 63.6], [23.8, 79.2], [18.9, 7.6], [86.1, 43.1], [28.9, 48.2]]



// Load the SVG file names from the text file
fetch("js/svg_paths.txt")
    .then(response => response.text())
    .then(data => {
        // Split the text file contents into an array of image names
        imageNames = data.trim().split("\n");
        // shuffleArray(imageNames)
        totalImages = imageNames.length;
        random_indices = [0, 6, 12, 18, 24, 30, 36]
        block_indices = [5, 11, 17, 23, 29, 35]

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
            // if (random_indices.includes(i)) {
            //     imageContainers += `
            //     <div id="image-container-${j + 1}" class="image-container">
            //         <table style="width:12cm;height:6cm;">
            //             <tr>
            //                 <th scope="col" style="font-weight: bold;">${(table_nums[table_i])[0]}</th>
            //                 <th scope="col" style="font-weight: bold;">${(table_nums[table_i])[1]}</th>
            //             </tr>

            //         </table>
            //     </div>
            //     `;
            //     table_i += 1
            //     j += 1
            // }

            if (block_indices.includes(i)){
                imageContainers += 
                `
                <div id="image-container-${j + 1}" class="image-container">
                    <h1>Block ${(i + 1) / 6} End.</h1>
                    <p> Continue when you are ready for the next 6 trials </p>
                </div>
                `;
                j += 1;
            }

        }


        // Add the image containers to the DOM
        document.getElementById("image-container").innerHTML = imageContainers;
        document.getElementById(`image-container-${currentImage}`).style.display = "block";

        // totalImages += 14
        totalImages += 6

        // Update the progress bar
        const progress = (currentImage / totalImages) * 100;
        document.querySelector('.progress').style.width = `${progress}%`;
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