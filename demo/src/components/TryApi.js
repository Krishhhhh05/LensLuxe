import React, { useState } from "react";
import Try from "../assets/try.jpg";

const TryApi = () => {
  const [message, setMessage] = useState("");
  const [updated, setUpdated] = useState(message);
  const handleChange = (event) => {
    setMessage(event.target.value);
  };
  console.log(message);
  const handleClick = () => {
    // 👇 "message" stores input field value
    setUpdated(message);
    console.log(message);
  // fetch(`http://localhost:5000/predict/?url=${encodeURIComponent(message)}`)
  // .then(response => response.json())

  fetch("http://localhost:5000/predict?url=" + encodeURIComponent(message), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      
    },
    body: JSON.stringify({}),
  })  .then(data => {
    console.log(data);
    // Do something with the response data
  })
  .catch(error => console.error(error));
    setMessage("");
    // .then((response) => response.json())
    // .then((data) => setPrediction(data.prediction))
    // .catch((error) => console.error(error));
  };

  return (
    <>
      <section class="text-gray-700 body-font border-t border-gray-200">
        <h2 className="font-dm font-bold text-3xl text-center lg:mt-8">
          Detect Your Image Here
        </h2>

        <div class="container lg:px-5 lg:py-8 mx-auto flex flex-wrap">
          <div class="lg:w-1/2 w-full mb-10 lg:mb-0 rounded-lg overflow-hidden">
            <img
              alt="feature"
              class="object-cover object-center h-full w-full"
              src={Try}
            />
          </div>
          <div class="flex flex-col flex-wrap lg:py-6 -mb-10 lg:w-1/2 lg:pl-12 lg:text-left text-center">
            <div class="flex flex-col mb-10 lg:items-start items-center lg:ml-12">
              <div class="flex-grow">
                <div class="w-12 h-12 inline-flex items-center justify-center rounded-full bg-[#008080] text-white mb-5">
                  <svg
                    fill="none"
                    stroke="currentColor"
                    stroke-linecap="round"
                    stroke-linejoin="round"
                    stroke-width="2"
                    class="w-6 h-6"
                    viewBox="0 0 24 24"
                  >
                    <path d="M22 12h-4l-3 9L9 3l-3 9H2"></path>
                  </svg>
                </div>
                <h2 class="text-gray-900 text-lg title-font font-medium mb-3">
                  Enter Your URL:
                </h2>
                <input
                  name="message"
                  type="email"
                  id="message"
                  value={message}
                  aria-describedby="helper-text-explanation"
                  class="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-full p-2.5  dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500"
                  onChange={handleChange}
                  placeholder="URL"
                />
                <p
                  id="helper-text-explanation"
                  class="mt-2 text-sm text-gray-500 dark:text-gray-400"
                >
                  We’ll never share your details. Read our{" "}
                  <a
                    href="#"
                    class="font-medium text-[#008080] hover:underline dark:[#008080]"
                  >
                    Privacy Policy
                  </a>
                  .
                </p>
              </div>
            </div>
            <button
              class="text-center inline-flex text-black bg-white py-2 px-6 focus:outline-none hover:bg-[#ff7f50] rounded text-lg font-dm border-black border-2 font-medium"
              onClick={handleClick}
            >
              View the Results
              <svg
                fill="none"
                stroke="currentColor"
                stroke-linecap="round"
                stroke-linejoin="round"
                stroke-width="2"
                class="w-4 h-4 ml-auto"
                viewBox="0 0 24 24"
              >
                <path d="M5 12h14M12 5l7 7-7 7"></path>
              </svg>
            </button>
          </div>
        </div>
      </section>
    </>
  );
};



export default TryApi;
