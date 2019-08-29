#pragma once

#include <fstream>
#include <string>
#include <memory>

#include <cuda/misc.hpp>
#include <cuda/memory.hpp>
#include <utils/attribute.hpp>
#include <utils/concepts.hpp>

namespace vol
{
template <typename Voxel>
struct Volume;

template <typename Voxel>
struct VolumeBlock
{
private:
	struct Inner : NoCopy, NoMove
	{
		std::string _;
		cuda::Extent dim;
	};

public:
	const Voxel *data() const
	{
		return reinterpret_cast<const Voxel *>( _->_.c_str() );
	}
	cuda::MemoryView3D<Voxel> view() const
	{
		auto view_info = cuda::MemoryView2DInfo{}
						   .set_stride( this->_->dim.width * sizeof( Voxel ) )
						   .set_width( this->_->dim.width )
						   .set_height( this->_->dim.height );
		return cuda::MemoryView3D<Voxel>( const_cast<char *>( _->_.c_str() ),
										  view_info, this->_->dim );
	}

private:
	VolumeBlock( std::string &&_, cuda::Extent dim )
	{
		this->_->_ = std::move( _ );
		this->_->dim = dim;
	}

	std::shared_ptr<Inner> _ = std::make_shared<Inner>();
	friend struct Volume<Voxel>;
};

template <typename Voxel>
struct Volume
{
	static Volume from_raw( const std::string &file_name, cuda::Extent const &dim )
	{
		Volume vol( std::ifstream( file_name, std::ios::in | std::ios::binary ), dim );
		vol.cnt = 1;
		return std::move( vol );
	}

	VolumeBlock<Voxel> get_block( std::size_t idx )
	{
		std::string buffer;
		auto block_size = sizeof( Voxel ) * dim.size();
		buffer.resize( block_size );
		auto buffer_ptr = const_cast<char *>( buffer.c_str() );
		auto nread = _.seekg( offset + idx * block_size )
					   .read( buffer_ptr, block_size )
					   .gcount();
		if ( nread != block_size ) {
			throw std::runtime_error( "failed to read block" );
		}
		return VolumeBlock<Voxel>( std::move( buffer ), dim );
	}

	std::size_t block_count() const { return cnt; }

private:
	Volume( std::ifstream &&_, cuda::Extent const &dim ) :
	  _( std::move( _ ) ),
	  dim( dim ) {}

private:
	std::ifstream _;
	std::size_t cnt;
	std::size_t offset = 0;
	cuda::Extent dim;
};

}  // namespace vol
