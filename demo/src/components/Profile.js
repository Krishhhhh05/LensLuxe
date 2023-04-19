import React from 'react'

const Profile = () => {
    return (
        <>
            <div className='grid grid-cols-9 bg-gradient-to-b from-green-200 to-lime-50'>
            
                <div className='col-span-3'>
                    
                </div>
                <div className="col-span-3 rounded-3xl overflow-hidden shadow-xl  my-5 justify-center bg-[#008080]">
                    <img src="https://i.imgur.com/dYcYQ7E.png" class="w-full" />
                    <div class="flex justify-center -mt-8">
                        <img src="https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=facearea&facepad=2&w=256&h=256&q=80" class="rounded-full border-solid border-white border-2 -mt-3" />
                    </div>
                    <div class="text-center px-3 pb-6 pt-2">
                        <h3 class="text-white text-sm bold font-sans">Olivia Dunham</h3>
                        <p class="mt-2 font-sans font-light text-white">Ace subscription</p>
                    </div>
                    <div class="flex justify-center pb-3 text-white">
                        <div class="text-center mr-3 border-r pr-3">
                            <h2>34</h2>
                            <span>Photos</span>
                        </div>
                        <div class="text-center">
                            <h2>42</h2>
                            <span>Friends</span>
                        </div>
                    </div>
                </div>
                <div className='col-span-3'>
                </div>
            </div>
        </>
    )
}

export default Profile